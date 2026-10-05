from unittest.mock import Mock

import pytest
import torch

from scripts import eval_latency


def test_synchronize_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("cuda", device)))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append(("mps", None)))

    eval_latency._synchronize(torch.device("cpu"))
    eval_latency._synchronize(torch.device("cuda:1"))
    eval_latency._synchronize(torch.device("mps"))

    assert calls == [("cuda", torch.device("cuda:1")), ("mps", None)]


@pytest.mark.parametrize(
    ("scope", "expected_events"),
    [
        ("extractor", ["forward", "sync", "clock", "extract", "sync", "clock"]),
        ("end-to-end", ["sync", "clock", "forward", "extract", "sync", "clock"]),
    ],
)
def test_timing_scope(scope, expected_events, monkeypatch):
    events = []

    def forward(_input):
        events.append("forward")
        return torch.tensor([[0.0, 1.0]])

    def extractor(_class_idx, _scores, **_kwargs):
        events.append("extract")
        return [torch.ones((1, 2, 2))]

    ticks = iter((1.0, 1.25))
    monkeypatch.setattr(eval_latency, "_synchronize", lambda _device: events.append("sync"))
    monkeypatch.setattr(eval_latency.time, "perf_counter", lambda: (events.append("clock"), next(ticks))[1])

    eval_latency._time_sample(
        Mock(side_effect=forward),
        torch.ones((1, 3, 2, 2)),
        extractor,
        None,
        torch.device("cpu"),
        scope,
        {},
    )

    assert events == expected_events


def test_validate_cams_rejects_invalid_output():
    with pytest.raises(RuntimeError):
        eval_latency._validate_cams([torch.tensor([[[torch.nan]]])], None)
    with pytest.raises(RuntimeError):
        eval_latency._validate_cams([torch.ones((1, 3, 3))], ((1, 2, 2),))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("UnknownCAM",), "invalid choice"),
        (("GradCAM", "--it", "0"), "expected a positive integer"),
        (("ScoreCAM", "--batch-size", "0"), "expected a positive integer"),
    ],
)
def test_cli_rejects_invalid_values(argv, message, capsys):
    with pytest.raises(SystemExit):
        eval_latency._build_parser().parse_args(argv)
    assert message in capsys.readouterr().err


def test_first_call_and_warmup_are_excluded(monkeypatch, mock_img_model):
    monkeypatch.setattr(eval_latency, "get_model", lambda *_args, **_kwargs: mock_img_model)
    sample = Mock(side_effect=[(idx / 1000, [torch.ones((1, 2, 2))]) for idx in range(1, 7)])
    monkeypatch.setattr(eval_latency, "_time_sample", sample)
    args = eval_latency._build_parser().parse_args(["LayerCAM", "--weights", "none", "--device", "cpu"])
    args.target_layer, args.size, args.warmup, args.it = ["0.3"], 16, 2, 3
    result = eval_latency._evaluate(args)
    assert sample.call_count == 6
    assert result["first_ms"] == 1
    assert result["samples_ms"] == [4, 5, 6]
