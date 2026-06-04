"""pytest configuration for the aqa datasets test suite.

Registers the ``slow`` mark (used on tests that read the local Fitness-AQA archive
or write video files via torchvision) so pytest does not emit PytestUnknownMarkWarning
when running the OHP unit tests.

Use ``-m "not slow"`` to skip archive-dependent tests in CI / local runs.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests that read the local Fitness-AQA archive or perform heavy I/O",
    )
