r"""
Tests for graph_bench.config module.
"""

import pytest

from graph_bench.config import DEFAULT_SCALE, ENV_PREFIX, SCALES, get_env, get_scale


class TestScales:
    def test_scales_defined(self):
        assert "small" in SCALES
        assert "medium" in SCALES
        assert "large" in SCALES

    def test_small_scale(self):
        # The aliases point at the LDBC SNB scale factors (see graph_bench/config.py).
        scale = SCALES["small"]
        assert scale is SCALES["sf01"]
        assert scale.nodes == 1_000
        assert scale.edges == 18_000

    def test_medium_scale(self):
        scale = SCALES["medium"]
        assert scale is SCALES["sf1"]
        assert scale.nodes == 10_000
        assert scale.edges == 180_000

    def test_large_scale(self):
        scale = SCALES["large"]
        assert scale is SCALES["sf100"]
        assert scale.nodes == 280_000
        assert scale.edges == 18_000_000

    def test_default_scale(self):
        assert DEFAULT_SCALE == "sf1"


class TestGetScale:
    def test_get_valid_scale(self):
        assert get_scale("small").name == "sf01"
        assert get_scale("sf3").name == "sf3"

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
