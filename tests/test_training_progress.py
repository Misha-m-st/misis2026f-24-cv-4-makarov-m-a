import json
import os
import sys
from types import ModuleType

import pytest

from document_vos.parallel import Progress
from document_vos.training_progress import MARKER, install_metrics_output, run_with_progress


def test_export_progress_prints_only_endpoints(monkeypatch, capsys):
    clock = iter(range(0, 100, 10))
    monkeypatch.setattr("document_vos.parallel.monotonic", lambda: next(clock))
    progress = Progress("Проверка", 3, 4, True, endpoints_only=True)
    for _ in range(3):
        progress.advance()
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "0/3" in lines[0] and "3/3" in lines[1]


@pytest.mark.parametrize("returncode", [0, 7])
def test_subprocess_full_log_compact_terminal(tmp_path, capsys, returncode):
    code = (
        "import sys; print('verbose model details'); "
        f"print({(MARKER + json.dumps({'iteration': 5, 'loss': 0.25, 'lr': 1e-5}))!r}); "
        f"sys.exit({returncode})"
    )
    path = tmp_path / "training.log"
    with path.open("w") as log:
        assert (
            run_with_progress([sys.executable, "-c", code], os.environ.copy(), log, 20)
            == returncode
        )
    terminal = capsys.readouterr()
    assert terminal.out == ""
    assert "verbose model details" not in terminal.err
    assert "loss=0.2500" in terminal.err
    assert "verbose model details" in path.read_text()
    assert MARKER in path.read_text()
    if returncode:
        assert "100%" not in terminal.err
        assert "ошибка" in terminal.err
    else:
        assert "100%" in terminal.err


def test_metrics_observer_preserves_official_logger(monkeypatch, capsys):
    calls = []

    class Logger:
        def log_metrics(self, *args):
            calls.append(args)

    module = ModuleType("cutie.utils.logger")
    module.TensorboardLogger = Logger
    monkeypatch.setitem(sys.modules, "cutie.utils.logger", module)
    install_metrics_output()
    metrics = {"total_loss": 0.125, "lr": 1e-5, "auxiliary_loss": 0.1}
    Logger().log_metrics("documents", "main_training", metrics, 25)
    assert calls == [("documents", "main_training", metrics, 25)]
    output = capsys.readouterr().out
    assert output.startswith(MARKER)
    assert json.loads(output[len(MARKER) :]) == {"iteration": 25, "loss": 0.125, "lr": 1e-5}
