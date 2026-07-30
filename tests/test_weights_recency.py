"""
recency_factor tests: it's a documented decay function over config data
(RECENCY_HALF_LIFE_DAYS, RECENCY_FLOOR), never computed live per-run and
never a guess - a claim with no observed_date is never discounted, a recent
one barely is, and an old one is discounted materially but never to zero.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.config.weights import RECENCY_FLOOR, recency_factor


def test_recency_factor_is_one_when_there_is_no_observed_date():
    assert recency_factor(None) == 1.0


def test_recency_factor_is_near_one_for_a_recent_date():
    as_of = date(2026, 7, 29)
    observed = as_of - timedelta(days=7)
    assert recency_factor(observed, as_of=as_of) > 0.95


def test_recency_factor_is_materially_lower_for_a_2019_date():
    as_of = date(2026, 7, 29)
    observed = date(2019, 1, 1)
    assert recency_factor(observed, as_of=as_of) < 0.3


def test_recency_factor_never_decays_to_zero():
    as_of = date(2026, 7, 29)
    ancient = date(1990, 1, 1)
    assert recency_factor(ancient, as_of=as_of) >= RECENCY_FLOOR


def test_recency_factor_decreases_monotonically_with_age():
    as_of = date(2026, 7, 29)
    recent = recency_factor(as_of - timedelta(days=30), as_of=as_of)
    older = recency_factor(as_of - timedelta(days=365), as_of=as_of)
    oldest = recency_factor(as_of - timedelta(days=365 * 5), as_of=as_of)
    assert recent > older > oldest


def test_recency_factor_treats_same_day_and_future_dates_as_current():
    as_of = date(2026, 7, 29)
    assert recency_factor(as_of, as_of=as_of) == 1.0
    assert recency_factor(as_of + timedelta(days=1), as_of=as_of) == 1.0
