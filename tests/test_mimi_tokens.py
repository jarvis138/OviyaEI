import os
import pytest


def test_mimi_import_available():
    try:
        import mimi  # type: ignore
    except Exception:
        pytest.skip("Mimi not available in CI")




