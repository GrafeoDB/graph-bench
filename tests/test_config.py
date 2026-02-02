r"""
Tests for graph_bench.config module.
"""

import os

import pytest

from graph_bench.config import DEFAULT_SCALE, ENV_PREFIX, SCALES, get_env, get_scale


class TestScales:
    def test_scales_defined(self):
        assert "small" in SCALES
        assert "medium" in SCALES
        assert "large" in SCALES

    def test_small_scale(self):
        scale = SCALES["small"]
        assert scale.name == "small"
        assert scale.nodes == 10_000
        assert scale.edges == 50_000

    def test_medium_scale(self):
        scale = SCALES["medium"]
        assert scale.name == "medium"
        assert scale.nodes == 100_000
        assert scale.edges == 500_000

    def test_large_scale(self):
        scale = SCALES["large"]
        assert scale.name == "large"
        assert scale.nodes == 1_000_000
        assert scale.edges == 5_000_000

    def test_default_scale(self):
        assert DEFAULT_SCALE == "medium"


class TestGetScale:
    def test_get_valid_scale(self):
        scale = get_scale("small")
        assert scale.name == "small"

    def test_get_invalid_scale(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            get_scale("invalid")


class TestGetEnv:
    def test_get_env_not_set(self):
        result = get_env("TEST_NOT_SET")
        assert result is None

    def test_get_env_with_default(self):
        result = get_env("TEST_NOT_SET", default="default_value")
        assert result == "default_value"

    def test_get_env_set(self, monkeypatch):
        monkeypatch.setenv(f"{ENV_PREFIX}TEST_VAR", "test_value")
        result = get_env("TEST_VAR")
        assert result == "test_value"

    def test_env_prefix(self):
        assert ENV_PREFIX == "GRAPH_BENCH_"
