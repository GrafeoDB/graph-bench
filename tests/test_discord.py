"""The Discord post: payload shape and sizes, the notify command and its HTTP handling (no network)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph_bench import notify
from graph_bench.cli import app
from graph_bench.release import is_newer, main, newest_stable
from graph_bench.report.discord import CONTENT_LIMIT, render_discord
from graph_bench.results import Results
from graph_bench.types import CATEGORY_ORDER, Category
from tests.test_report import _entry, _info

# An obviously fake placeholder, never a real webhook.
FAKE_URL = "https://example.invalid/webhook/fake-id/fake-token"
DBS = [f"db{i:02d}" for i in range(12)]


def _results(entries: list, names: list[str], version: str = "0.6.2") -> Results:
    return Results(
        graph_bench_commit="0123456789abcdef",
        grafeo={"version": version, "source": "pypi", "sha": "abcdef1234567890"},
        machine={"cpu": "Test CPU", "cores": 8, "os": "TestOS 1"},
        selection="full",
        mode="bench",
        started_at="2026-10-31T06:00:00Z",
        databases={name: _info(name) for name in names},
        entries=entries,
    )


@pytest.fixture
def big() -> Results:
    """Every category with 13 databases and 12 benchmarks each: enough to need several messages."""
    entries = []
    for category in CATEGORY_ORDER:
        cat: Category = category
        for b in range(12):
            for i, db in enumerate(["grafeo", *DBS]):
                failed = (i + b) % 7 == 0 and db != "grafeo"
                entries.append(
                    _entry(
                        cat,
                        f"{cat}_bench_{b:02d}",
                        db,
                        "failed" if failed else "ok",
                        "unchecked" if failed else "agree",
                        None if failed else 1.0 + i + b / 10,
                    )
                )
    return _results(entries, ["grafeo", *DBS])


@pytest.fixture
def small() -> Results:
    entries = [
        _entry("algorithms", "graphalytics_bfs", "grafeo", "ok", "agree", 2.0),
        _entry("algorithms", "graphalytics_bfs", "ladybug", "ok", "agree", 1.5),
    ]
    return _results(entries, ["grafeo", "ladybug"], version="0.6.1")


def test_payloads_fit_discord_limits(big: Results) -> None:
    messages = render_discord(big, run_url="https://example.invalid/run/1")
    assert len(messages) >= 3
    for message in messages:
        assert 0 < len(message["content"]) <= CONTENT_LIMIT
        assert message["content"].count("```") % 2 == 0
        assert message["allowed_mentions"] == {"parse": []}
        assert message["flags"] == 4
    text = "\n".join(m["content"] for m in messages)
    assert text.count("Complex queries\n") == 1


def test_first_message_has_trigger_and_links(big: Results) -> None:
    messages = render_discord(big, run_url="https://example.invalid/run/1", readme_url="https://example.invalid/readme")
    first = messages[0]["content"].splitlines()
    assert first[0] == "**Grafeo 0.6.2 released**"
    assert "CI run: https://example.invalid/run/1" in first
    assert "Results: https://example.invalid/readme" in first


def test_summary_is_last_and_names_only_grafeo(big: Results) -> None:
    messages = render_discord(big)
    last = messages[-1]["content"]
    assert last.startswith("```\nGrafeo fastest")
    assert "db01" not in last
    assert "CI run" not in "\n".join(m["content"] for m in messages)


def test_small_results_use_three_messages_and_the_cli_prints_json(small: Results, tmp_path: Path) -> None:
    assert len(render_discord(small)) == 3
    path = tmp_path / "results.json"
    small.write(path)
    args = ["report", "--format", "discord", "--in", str(path), "--run-url", "https://example.invalid/r"]
    out = CliRunner().invoke(app, args)
    assert out.exit_code == 0, out.output
    payloads = json.loads(out.output)
    assert payloads[0]["content"].startswith("**Grafeo 0.6.1 released**")
    assert "CI run: https://example.invalid/r" in payloads[0]["content"]


def test_notify_skips_without_the_variable(small: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "results.json"
    small.write(path)
    monkeypatch.delenv("FAKE_HOOK", raising=False)
    monkeypatch.setattr(notify, "urllib_send", lambda *a: pytest.fail("no request may be made"))
    out = CliRunner().invoke(app, ["notify", "--in", str(path), "--webhook-env", "FAKE_HOOK"])
    assert out.exit_code == 0
    assert "FAKE_HOOK is not set; skipping" in out.output


def test_notify_posts_every_message_and_never_prints_the_url(
    big: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.json"
    big.write(path)
    sent: list[tuple[str, bytes]] = []

    def fake_send(url: str, body: bytes) -> tuple[int, str]:
        sent.append((url, body))
        return 204, ""

    monkeypatch.setenv("FAKE_HOOK", FAKE_URL)
    monkeypatch.setattr(notify, "urllib_send", fake_send)
    monkeypatch.setattr(notify.time, "sleep", lambda s: None)
    out = CliRunner().invoke(app, ["notify", "--in", str(path), "--webhook-env", "FAKE_HOOK"])
    assert out.exit_code == 0, out.output
    assert len(sent) == len(render_discord(big))
    assert all(url == FAKE_URL for url, _ in sent)
    assert FAKE_URL not in out.output
    assert "fake-token" not in out.output


def test_429_is_retried_after_retry_after() -> None:
    replies = iter([(429, json.dumps({"retry_after": 1.5})), (429, "not json"), (204, "")])
    waits: list[float] = []
    count = notify.post_all(FAKE_URL, [{"content": "x"}], send=lambda u, b: next(replies), sleep=waits.append)
    assert count == 1
    assert waits == [1.5, 1.0]


def test_429_gives_up_after_the_retry_limit() -> None:
    with pytest.raises(notify.NotifyError, match="HTTP 429"):
        notify.post_all(
            FAKE_URL, [{"content": "x"}], send=lambda u, b: (429, '{"retry_after": 0}'), sleep=lambda s: None
        )


def test_other_failures_carry_the_body_but_not_the_url() -> None:
    body = f"bad request for {FAKE_URL}"
    with pytest.raises(notify.NotifyError) as err:
        notify.post_all(FAKE_URL, [{"content": "x"}], send=lambda u, b: (400, body), sleep=lambda s: None)
    assert "HTTP 400" in str(err.value)
    assert "bad request" in str(err.value)
    assert FAKE_URL not in str(err.value)


def test_transport_errors_do_not_leak_the_url() -> None:
    def boom(url: str, body: bytes) -> tuple[int, str]:
        raise OSError(f"cannot reach {url}")

    with pytest.raises(notify.NotifyError) as err:
        notify.post_all(FAKE_URL, [{"content": "x"}], send=boom, sleep=lambda s: None)
    assert FAKE_URL not in str(err.value)


def test_notify_reports_failure_with_exit_1(small: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "results.json"
    small.write(path)
    monkeypatch.setenv("FAKE_HOOK", FAKE_URL)
    monkeypatch.setattr(notify, "urllib_send", lambda u, b: (500, "server error"))
    monkeypatch.setattr(notify.time, "sleep", lambda s: None)
    out = CliRunner().invoke(app, ["notify", "--in", str(path), "--webhook-env", "FAKE_HOOK"])
    assert out.exit_code == 1
    assert "HTTP 500" in out.output
    assert FAKE_URL not in out.output


def test_notify_bad_results_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HOOK", FAKE_URL)
    out = CliRunner().invoke(app, ["notify", "--in", str(tmp_path / "none.json"), "--webhook-env", "FAKE_HOOK"])
    assert out.exit_code == 2
    assert FAKE_URL not in out.output


def test_newest_stable_ignores_prereleases_and_yanked() -> None:
    live = [{"yanked": False}]
    releases = {
        "0.5.9": live,
        "0.6.0": live,
        "0.6.2rc1": live,
        "0.7.0.dev1": live,
        "0.6.10": live,
        "0.6.11": [{"yanked": True}],
        "0.8.0": [],
    }
    assert newest_stable(releases) == "0.6.10"
    assert newest_stable({"1.0.0a1": live}) is None


def test_is_newer_orders_numerically() -> None:
    assert is_newer("0.6.10", "0.6.9")  # newer posts
    assert is_newer("0.7.0", "0.6.12")
    assert is_newer("0.6.1", "")  # never posted
    assert not is_newer("0.6.2", "0.6.2")  # equal skips
    assert not is_newer("0.6", "0.6.0")  # trailing zeros are the same version
    assert not is_newer("0.6.1", "0.6.2")  # an older newest (a yanked release) skips
    assert not is_newer("0.6.9", "0.6.10")  # string order would say newer
    with pytest.raises(ValueError):
        is_newer("0.7.0rc1", "0.6.2")  # pre-releases are not versions to post
    with pytest.raises(ValueError):
        is_newer("0.7.0", "garbage")


def test_release_newer_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["newer", "0.6.10", "0.6.9"]) == 0
    assert main(["newer", "0.6.2", "0.6.2"]) == 1
    assert main(["newer", "0.6.1", "0.6.2"]) == 1
    assert main(["newer", "0.6.1", ""]) == 0
    assert main(["newer", "0.7.0rc1", "0.6.2"]) == 2
    assert main(["bogus"]) == 2
    capsys.readouterr()


def test_prereleases_are_not_picked_as_newest() -> None:
    live = [{"yanked": False}]
    newest = newest_stable({"0.6.2": live, "0.7.0rc1": live, "0.7.0.dev3": live})
    assert newest == "0.6.2"
    assert not is_newer(newest, "0.6.2")


def test_transport_errors_are_retried_with_backoff() -> None:
    calls = {"n": 0}
    waits: list[float] = []

    def flaky(url: str, body: bytes) -> tuple[int, str]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("blip")
        return 204, ""

    assert notify.post_all(FAKE_URL, [{"content": "x"}], send=flaky, sleep=waits.append) == 1
    assert calls["n"] == 3
    assert waits == [2.0, 4.0]


def test_resume_skips_messages_already_sent() -> None:
    sent: list[str] = []
    progress: list[int] = []

    def send(url: str, body: bytes) -> tuple[int, str]:
        sent.append(json.loads(body)["content"])
        return 204, ""

    payloads = [{"content": c} for c in "abcd"]
    notify.post_all(FAKE_URL, payloads, start=2, on_sent=progress.append, send=send, sleep=lambda s: None)
    assert sent == ["c", "d"]
    assert progress == [3, 4]


def test_cli_progress_file_resumes_after_a_partial_failure(
    big: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.json"
    big.write(path)
    progress = tmp_path / "progress.txt"
    total = len(render_discord(big))
    sent: list[bytes] = []
    fail_at = {"n": 2}

    def send(url: str, body: bytes) -> tuple[int, str]:
        if len(sent) == fail_at["n"]:
            return 500, "boom"
        sent.append(body)
        return 204, ""

    monkeypatch.setenv("FAKE_HOOK", FAKE_URL)
    monkeypatch.setattr(notify, "urllib_send", send)
    monkeypatch.setattr(notify.time, "sleep", lambda s: None)
    args = ["notify", "--in", str(path), "--webhook-env", "FAKE_HOOK", "--progress", str(progress)]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 1
    assert f"2 of {total} messages had gone out" in first.output
    assert progress.read_text(encoding="utf-8").strip() == "2"
    fail_at["n"] = -1
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0, second.output
    assert len(sent) == total  # each message exactly once across both attempts
    assert len(set(sent)) == total
    assert FAKE_URL not in first.output + second.output
