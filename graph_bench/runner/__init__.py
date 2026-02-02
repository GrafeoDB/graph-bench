r"""
Benchmark runner and orchestration.

Coordinates benchmark execution, timing, warmup,
and result collection across multiple adapters.

    from graph_bench.runner import BenchmarkOrchestrator

    orchestrator = BenchmarkOrchestrator()
    results = orchestrator.run(adapters, benchmarks, scale="medium")
"""

from graph_bench.runner.orchestrator import BenchmarkOrchestrator, OrchestratorConfig, OrchestratorResult
from graph_bench.runner.timing import Timer, measure_iterations, measure_time

__all__ = [
    "BenchmarkOrchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    "Timer",
    "measure_iterations",
    "measure_time",
]
