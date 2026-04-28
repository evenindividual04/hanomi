"""Tests for calibration metrics."""

import pytest
import numpy as np
from src.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    compute_calibration_metrics,
)


def test_brier_score():
    """Test Brier score calculation."""
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.3, 0.2]

    score = brier_score(y_true, y_prob)

    # Perfect predictions would be 0.0
    # These are good predictions, so score should be low
    assert 0.0 <= score <= 1.0
    assert score < 0.1  # Should be quite good


def test_brier_score_perfect():
    """Test Brier score with perfect predictions."""
    y_true = [1, 1, 0, 0]
    y_prob = [1.0, 1.0, 0.0, 0.0]

    score = brier_score(y_true, y_prob)

    assert score == 0.0


def test_brier_score_worst():
    """Test Brier score with worst predictions."""
    y_true = [1, 1, 0, 0]
    y_prob = [0.0, 0.0, 1.0, 1.0]  # Completely wrong

    score = brier_score(y_true, y_prob)

    assert score == 1.0


def test_expected_calibration_error():
    """Test ECE calculation."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.2, 0.1, 0.3]

    ece = expected_calibration_error(y_true, y_prob, n_bins=3)

    assert 0.0 <= ece <= 1.0


def test_expected_calibration_error_perfect():
    """Test ECE with perfect calibration (confidence = accuracy per bin)."""
    # With 10 bins, these predictions land in the 0.9-1.0 bin (acc=1.0, conf≈0.95)
    # and 0.0-0.1 bin (acc=0.0, conf≈0.05), giving near-zero ECE.
    y_true = [1, 1, 0, 0]
    y_prob = [0.95, 0.95, 0.05, 0.05]

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    assert ece < 0.1


def test_compute_calibration_metrics():
    """Test full calibration metrics computation."""
    results = [
        {"confidence": 0.9, "is_true_positive": True},
        {"confidence": 0.8, "is_true_positive": True},
        {"confidence": 0.7, "is_true_positive": False},
        {"confidence": 0.2, "is_true_positive": False},
    ]

    metrics = compute_calibration_metrics(results)

    assert "brier_score" in metrics
    assert "ece" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_compute_calibration_metrics_perfect():
    """Test calibration metrics with truly perfect predictions (confidence ∈ {0, 1})."""
    results = [
        {"confidence": 1.0, "is_true_positive": True},
        {"confidence": 1.0, "is_true_positive": True},
        {"confidence": 0.0, "is_true_positive": False},
        {"confidence": 0.0, "is_true_positive": False},
    ]

    metrics = compute_calibration_metrics(results)

    assert metrics["brier_score"] == 0.0
    assert metrics["ece"] == 0.0


def test_compute_calibration_metrics_empty():
    """Test calibration metrics with empty results."""
    results = []

    metrics = compute_calibration_metrics(results)

    assert "brier_score" in metrics
    assert "ece" in metrics
    assert metrics["brier_score"] == 0.0
    assert metrics["ece"] == 0.0
