import subprocess
import sys

from document_vos.diagnostics import parser


def test_cli_split_validate_export_report(tmp_path, manifest):
    def command(*args):
        result = subprocess.run(
            [sys.executable, "-m", "document_vos", "tools", *map(str, args)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert '"frames": 24' in command("validate-data", "--manifest", manifest)
    split = tmp_path / "split"
    command("split", "--manifest", manifest, "--output", split)
    command(
        "export-training",
        "--manifest",
        split / "train.jsonl",
        "--split-dir",
        split,
        "--output",
        tmp_path / "export",
        "--workers",
        "2",
    )
    assert (tmp_path / "export/export.json").is_file()
    bad = subprocess.run(
        [sys.executable, "-m", "document_vos", "tools", "train"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode == 2 and "Необходимо" in bad.stderr


def test_all_nine_commands_expose_config():
    for name in [
        "prepare",
        "split",
        "validate-data",
        "export-training",
        "train",
        "infer",
        "evaluate",
        "compare",
        "report",
    ]:
        assert (
            parser().parse_args([name, "--config", "settings.yaml"]).config.name == "settings.yaml"
        )
