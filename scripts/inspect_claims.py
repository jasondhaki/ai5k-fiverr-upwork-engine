"""
Manual audit tool for the ingestion stage ONLY.

Runs extract_claims over a CV and/or a GitHub username and prints every
resulting claim in a form a human can check against its source: does the
span actually contain what claim_text asserts? For every grounded claim it
also re-runs reverify_span against the currently stored text document, so a
stale or corrupted document_id shows up here instead of silently downstream.

No scoring, no gap ranking, no asset generation - this is ingestion output,
inspected raw, nothing else.

Usage:
    python scripts/inspect_claims.py --cv path/to/resume.pdf --github octocat
    python scripts/inspect_claims.py --cv path/to/resume.pdf
    python scripts/inspect_claims.py --github octocat
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Extracted spans can contain arbitrary Unicode (READMEs and CVs use en-dashes,
# arrows, etc.) but Windows consoles default to a codepage (cp1252) that can't
# encode all of it. Reconfigure stdout to UTF-8 so audit output never crashes
# on a claim's content rather than showing it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.ingestion.span_grounding import reverify_span  # noqa: E402
from app.platform.pipeline import PipelineInput, extract_claims  # noqa: E402
from app.schemas import Claim  # noqa: E402
from app.storage.store import file_store  # noqa: E402


def _reverify(claim: Claim) -> str:
    """Re-read the span's stored text document right now and confirm the
    span still resolves to the same substring - "PASS"/"FAIL" mirror what the
    generation-time validator would decide if it ran this instant."""
    span = claim.source_span
    if span is None:
        return "n/a (ungrounded)"
    try:
        current_text = file_store.get_text(span.document_id)
    except FileNotFoundError:
        return "FAIL (text document missing)"
    return "PASS" if reverify_span(span, current_text) else "FAIL"


def _format_claim(claim: Claim) -> str:
    lines = [
        f"[{claim.evidence_tier.value}] weight={claim.weight:.2f} "
        f"recency={claim.recency_factor:.2f} publishable={claim.publishable}",
        f"claim: {claim.claim_text}",
    ]

    span = claim.source_span
    if span is None:
        lines.append(f"source: {claim.source_type.value} / (ungrounded - no source span)")
        lines.append("span  : (none - dropped to a coaching prompt, not published)")
    else:
        locator = span.locator or "n/a"
        doc_short = span.document_id[:8]
        lines.append(f"source: {claim.source_type.value} / {locator} / doc {doc_short}")
        lines.append(f'span  : "{span.text}"')

    lines.append(f"reverify: {_reverify(claim)}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestion only (extract_claims) and print every claim for manual audit."
    )
    parser.add_argument("--cv", type=Path, help="Path to a native-text PDF resume")
    parser.add_argument("--github", type=str, help="GitHub username")
    args = parser.parse_args()

    if not args.cv and not args.github:
        parser.error("provide at least one of --cv or --github")
    if args.cv and not args.cv.is_file():
        parser.error(f"no such file: {args.cv}")

    return args


def main() -> None:
    # Ingestion runs silently otherwise - with no output between startup and
    # the final claim dump, a provider retrying under rate-limit backoff
    # (which has taken over 20 minutes in practice) looks identical to a
    # hang. This surfaces the progress logging already emitted by the
    # parsers/extractor (which source is starting, which pass is retrying
    # and why) instead of leaving the terminal blank while it waits.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    args = _parse_args()
    cv_bytes = args.cv.read_bytes() if args.cv else None

    inp = PipelineInput(
        niche="cli-inspect",  # unused by extract_claims; PipelineInput just requires it
        cv_bytes=cv_bytes,
        github_username=args.github,
    )

    try:
        claims = extract_claims(inp)
    except ValueError as exc:
        print(f"ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    grouped: dict[str, list[Claim]] = {}
    for claim in claims:
        grouped.setdefault(claim.source_type.value, []).append(claim)

    for source_type, source_claims in grouped.items():
        print(f"=== {source_type} ({len(source_claims)} claims) ===")
        for claim in source_claims:
            print(_format_claim(claim))
            print("-" * 60)
        print()

    total = len(claims)
    publishable = sum(1 for c in claims if c.publishable)
    ungrounded = total - publishable
    per_source = ", ".join(f"{k}={len(v)}" for k, v in grouped.items()) or "none"

    print("=== summary ===")
    print(
        f"total={total} publishable={publishable} ungrounded={ungrounded} "
        f"per_source=[{per_source}]"
    )


if __name__ == "__main__":
    main()
