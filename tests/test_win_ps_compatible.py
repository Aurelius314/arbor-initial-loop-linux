"""Windows/PowerShell compatibility smoke checks for the local Arbor run."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
ARBOR = PROJECT.parent / "Arbor-ref"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-specific checks")


@pytest.fixture
def local_tmp() -> Path:
    path = PROJECT / ".win-compat-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)


def test_eval_command_runs_from_a_path_with_spaces(local_tmp: Path):
    """The protected command must not depend on Bash path parsing."""
    spaced = local_tmp / "project with spaces"
    spaced.mkdir()
    command = [sys.executable, str(PROJECT / "eval.py"), "--split", "dev", "--backend", "mock"]
    result = run(command, spaced)
    assert result.returncode == 0, result.stderr
    assert "score:" in result.stdout


def test_git_worktree_create_and_remove(local_tmp: Path):
    """Exercise the Git primitive used to isolate Arbor experiments."""
    repo = local_tmp / "repo"
    worktree = local_tmp / "worktree with spaces"
    repo.mkdir()
    commands = [
        ["git", "init"],
        ["git", "config", "user.name", "Arbor Test"],
        ["git", "config", "user.email", "arbor-test@local"],
    ]
    for command in commands:
        assert run(command, repo).returncode == 0
    (repo / "baseline.txt").write_text("baseline", encoding="utf-8")
    assert run(["git", "add", "baseline.txt"], repo).returncode == 0
    assert run(["git", "commit", "-m", "baseline"], repo).returncode == 0
    created = run(["git", "worktree", "add", "-b", "arbor/win-smoke", str(worktree), "HEAD"], repo)
    assert created.returncode == 0, created.stderr
    assert (worktree / "baseline.txt").is_file()
    removed = run(["git", "worktree", "remove", "--force", str(worktree)], repo)
    assert removed.returncode == 0, removed.stderr


def test_main_arbor_user_suffix_is_windows_safe():
    """The main worktree implementation must guard the POSIX-only getuid API."""
    source = (ARBOR / "src/coordinator/tools/git_ops.py").read_text(encoding="utf-8")
    assert 'hasattr(os, "getuid")' in source
    session_ops = (ARBOR / "src/mcp/session_ops.py").read_text(encoding="utf-8")
    assert "getpass.getuser" in session_ops or "_user_suffix" in session_ops


def test_protected_paths_use_post_run_verification_on_windows():
    """Windows read-only protection is best-effort, so hash verification is required."""
    config = (ARBOR / "src/coordinator/config.py").read_text(encoding="utf-8")
    executor = (ARBOR / "src/coordinator/tools/executor_run.py").read_text(encoding="utf-8")
    combined = config + executor
    assert "enforce_protected" in combined
    assert "protected" in combined.lower()
    assert "manifest" in combined.lower() or "hash" in combined.lower()


@pytest.mark.xfail(
    condition=not hasattr(os, "getuid"),
    reason="legacy skill helper still calls POSIX-only os.getuid(); main Arbor path is safe",
    strict=True,
)
def test_legacy_arbor_state_has_no_unconditional_getuid():
    source = (ARBOR / "skills/arbor-agent-tools/scripts/arbor_state.py").read_text(encoding="utf-8")
    assert "os.getuid()" not in source
