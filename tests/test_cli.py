from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

from click.testing import CliRunner
import pytest

from llmling.cli import ExitCode, cli
from llmling.task.models import TaskResult


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_task_result() -> TaskResult:
    """Create a mock task result."""
    return TaskResult(
        content="Test result",
        model="test-model",
        context_metadata={},
        completion_metadata={},
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary test configuration file."""
    config_content = """
version: "1.0"
global_settings:
  timeout: 30
  max_retries: 3
  temperature: 0.7

context_processors: {}

llm_providers:
  test-provider:
    name: Test Provider
    model: test/model
    temperature: 0.7

contexts:
  test-context:
    type: text
    content: "Test content"
    description: "Test context"

task_templates:
  test-template:
    provider: test-provider
    context: test-context
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(config_content)
    return config_path


def test_cli_without_config(runner: CliRunner) -> None:
    """Test CLI behavior without config."""
    result = runner.invoke(cli, ["validate"])
    assert "Missing option '-c'" in result.output
    assert result.exit_code == ExitCode.USAGE


def test_cli_invalid_config(runner: CliRunner, tmp_path: Path) -> None:
    """Test CLI behavior with invalid config."""
    # Create a syntactically valid but semantically invalid YAML
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text("""
version: "1.0"
invalid_key: true
""")

    result = runner.invoke(cli, ["-c", str(bad_config), "validate"])
    assert result.exit_code == ExitCode.ERROR
    assert "Configuration Error" in result.output


def test_cli_valid_config(runner: CliRunner, config_file: Path) -> None:
    """Test CLI behavior with valid config."""
    result = runner.invoke(cli, ["-c", str(config_file), "validate"])
    assert result.exit_code == 0
    assert "Configuration is valid!" in result.output


def test_cli_nonexistent_config(runner: CliRunner, tmp_path: Path) -> None:
    """Test CLI behavior with nonexistent config."""
    nonexistent = tmp_path / "nonexistent.yml"
    result = runner.invoke(cli, ["-c", str(nonexistent), "validate"])
    assert "does not exist" in result.output
    assert result.exit_code != 0


def test_cli_list_templates(runner: CliRunner, config_file: Path) -> None:
    """Test list-templates command."""
    result = runner.invoke(cli, ["-c", str(config_file), "list-templates"])
    assert result.exit_code == 0
    assert "Available Task Templates" in result.output


@pytest.mark.asyncio
async def test_cli_execute(
    runner: CliRunner,
    config_file: Path,
    mock_task_result: TaskResult,
) -> None:
    """Test execute command."""

    async def mock_execute(*args, **kwargs):
        return mock_task_result

    with mock.patch("llmling.task.manager.TaskManager.execute_template") as mock_exec:
        mock_exec.return_value = mock_task_result
        # Make the mock async
        mock_exec.side_effect = mock_execute

        result = runner.invoke(
            cli,
            ["-c", str(config_file), "execute", "test-template"],
            catch_exceptions=False,
        )

        print(f"Output: {result.output}")  # Debug output
        print(f"Exception: {result.exception}")  # Debug output

        assert result.exit_code == ExitCode.SUCCESS
        assert "Test result" in result.output


def test_cli_unknown_command(runner: CliRunner, config_file: Path) -> None:
    """Test behavior with unknown command."""
    result = runner.invoke(cli, ["-c", str(config_file), "unknown"])
    assert result.exit_code != 0
    assert "No such command" in result.output


if __name__ == "__main__":
    pytest.main(["-v", __file__])
