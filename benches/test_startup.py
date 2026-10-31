"""Startup: a fresh process opens an empty database and answers one probe query (`graph_bench.runner.startup`)."""

from benches._cases import make_startup_test

test_startup = make_startup_test()
