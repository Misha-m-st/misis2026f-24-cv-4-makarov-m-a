"""Компактная сводка в терминале; полный вывод subprocess остаётся в журнале."""

import json
import subprocess
import sys

MARKER = "DOCUMENT_VOS_PROGRESS "


def install_metrics_output() -> None:
    """Используем готовые усреднённые метрики официального logger, без нового loss."""
    from cutie.utils.logger import TensorboardLogger

    original = TensorboardLogger.log_metrics

    def log_metrics(self, exp_id, prefix, metrics, it):
        original(self, exp_id, prefix, metrics, it)
        print(
            MARKER
            + json.dumps(
                {"iteration": it, "loss": metrics.get("total_loss"), "lr": metrics.get("lr")}
            ),
            flush=True,
        )

    TensorboardLogger.log_metrics = log_metrics


def run_with_progress(command: list[str], env: dict, log, total: int) -> int:
    from tqdm import tqdm

    print(f"Обучение: запуск модели и загрузчиков; журнал: {log.name}", file=sys.stderr, flush=True)
    with tqdm(
        total=total, desc="Обучение", unit="ит", dynamic_ncols=True, mininterval=10, file=sys.stderr
    ) as bar:
        with subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(MARKER):
                    metrics = json.loads(line[len(MARKER) :])
                    postfix = {}
                    if metrics["loss"] is not None:
                        postfix["loss"] = f"{metrics['loss']:.4f}"
                    if metrics["lr"] is not None:
                        postfix["lr"] = f"{metrics['lr']:.1e}"
                    bar.set_postfix(postfix, refresh=False)
                    bar.update(metrics["iteration"] - bar.n)
            returncode = process.wait()
            if returncode == 0:
                bar.update(total - bar.n)
    print(
        "Обучение: процесс завершён"
        if returncode == 0
        else f"Обучение: ошибка (код {returncode}); подробности: {log.name}",
        file=sys.stderr,
        flush=True,
    )
    return returncode
