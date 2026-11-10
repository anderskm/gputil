import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import patch

from GPUtil.GPUtil import getFirstAvailable


def test_no_attempts_raises_runtime_error_not_unbound_local_error():
    # attempts=0 runs zero loop iterations, so no GPU can be found. The
    # documented contract is RuntimeError, not UnboundLocalError.
    with pytest.raises(RuntimeError, match="Could not find an available GPU"):
        getFirstAvailable(attempts=0, interval=0)


def test_exhausted_attempts_raise_runtime_error_without_gpu():
    # No GPUs detected, so one attempt finds nothing and must raise
    # the documented RuntimeError (mocked: no hardware required).
    with patch("GPUtil.GPUtil.getGPUs", return_value=[]):
        with pytest.raises(RuntimeError, match="Could not find an available GPU"):
            getFirstAvailable(attempts=1, interval=0)
