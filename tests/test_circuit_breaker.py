"""
Tests for CircuitBreaker — per-model failure tracking with cooldown window.

Verifies:
- Failure counting and threshold tripping
- Cooldown window enforcement
- Pruning of stale failures outside window
- Success resetting the circuit
- remaining_cooldown() precision
- open_models listing
"""

import time
from model_router.client import CircuitBreaker


class TestFailureCounting:
    def test_fresh_model_is_closed(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        assert cb.is_open("model-a") is False
        assert cb.remaining_cooldown("model-a") == 0.0

    def test_under_threshold_stays_closed(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is False

    def test_at_threshold_opens(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is True

    def test_above_threshold_stays_open(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        for _ in range(5):
            cb.record_failure("model-a")
        assert cb.is_open("model-a") is True

    def test_success_resets_counter(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        for _ in range(3):
            cb.record_failure("model-a")
        assert cb.is_open("model-a") is True

        cb.record_success("model-a")
        assert cb.is_open("model-a") is False
        assert cb.remaining_cooldown("model-a") == 0.0

    def test_success_only_resets_that_model(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        for _ in range(3):
            cb.record_failure("model-a")
            cb.record_failure("model-b")
        assert cb.is_open("model-a") is True
        assert cb.is_open("model-b") is True

        cb.record_success("model-a")
        assert cb.is_open("model-a") is False
        assert cb.is_open("model-b") is True


class TestCooldown:
    def test_remaining_cooldown_positive(self):
        cb = CircuitBreaker(cooldown=10, max_failures=2)
        cb.record_failure("model-a")
        cb.record_failure("model-a")  # triggers cooldown
        remaining = cb.remaining_cooldown("model-a")
        assert remaining > 0
        assert remaining <= 10

    def test_cooldown_expires(self):
        cb = CircuitBreaker(cooldown=0.05, max_failures=2)
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is True
        time.sleep(0.1)
        assert cb.is_open("model-a") is False
        assert cb.remaining_cooldown("model-a") == 0.0

    def test_failure_within_window_reopens(self):
        """New failures within the cooldown window keep circuit open."""
        cb = CircuitBreaker(cooldown=1, max_failures=2)
        # Trip the circuit
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is True

        # Wait a tiny bit, then fail again — circuit stays open
        time.sleep(0.05)
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is True

    def test_stale_failures_pruned(self):
        """Failures older than cooldown are pruned from the list."""
        cb = CircuitBreaker(cooldown=0.05, max_failures=2)
        cb.record_failure("model-a")
        time.sleep(0.06)
        # First failure is now stale; second should not trip (only 1 active)
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is False  # only 1 active failure < 2


class TestOpenModels:
    def test_open_models_empty_by_default(self):
        cb = CircuitBreaker(cooldown=60, max_failures=3)
        assert cb.open_models == []

    def test_open_models_returns_tripped_models(self):
        cb = CircuitBreaker(cooldown=60, max_failures=2)
        cb.record_failure("model-a")
        cb.record_failure("model-b")
        cb.record_failure("model-b")  # model-b trips
        assert "model-a" not in cb.open_models
        assert "model-b" in cb.open_models

    def test_open_models_excludes_recovered(self):
        cb = CircuitBreaker(cooldown=60, max_failures=2)
        cb.record_failure("model-b")
        cb.record_failure("model-b")  # trips
        assert "model-b" in cb.open_models
        cb.record_success("model-b")
        assert "model-b" not in cb.open_models


class TestIsolation:
    def test_different_models_independent(self):
        """Failures on one model don't affect another."""
        cb = CircuitBreaker(cooldown=60, max_failures=2)
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        assert cb.is_open("model-a") is True
        assert cb.is_open("model-b") is False

    def test_unknown_model_is_closed(self):
        cb = CircuitBreaker()
        assert cb.is_open("never-seen") is False

    def test_no_failures_returns_zero_cooldown(self):
        cb = CircuitBreaker()
        assert cb.remaining_cooldown("never-seen") == 0.0
