"""Fresh-holdout provenance must never enable head training."""

import subprocess
import sys


def test_fresh_holdout_role_prohibits_training_before_loading_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rejection_lab.train_existence",
            "--baseline-repo",
            str(tmp_path / "missing-repo"),
            "--model",
            str(tmp_path / "missing-model"),
            "--dataset",
            str(tmp_path / "missing-data"),
            "--output",
            str(tmp_path / "output"),
            "--evaluation-role",
            "fresh-synthetic-holdout",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "requires --head; training is prohibited" in result.stderr
    assert not (tmp_path / "output").exists()
