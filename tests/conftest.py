import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas._testing import assert_frame_equal, assert_series_equal
from scipy.spatial.transform import Rotation
from tpcp import BaseTpcpObject

from tests._regression_utils import PyTestSnapshotTest

try:
    from pomegranate import GeneralMixtureModel, State
except ImportError:
    GeneralMixtureModel = None
    State = None


@pytest.fixture(autouse=True)
def reset_random_seed() -> None:
    np.random.seed(10)
    random.seed(10)


@pytest.fixture()
def snapshot(request):
    with PyTestSnapshotTest(request) as snapshot_test:
        yield snapshot_test


def pytest_addoption(parser) -> None:
    group = parser.getgroup("snapshottest")
    group.addoption(
        "--snapshot-update", action="store_true", default=False, dest="snapshot_update", help="Update the snapshots."
    )
