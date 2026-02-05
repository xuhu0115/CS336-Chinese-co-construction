from typing import TypeVar
import numpy as np
import pytest
import os
from pathlib import Path
import torch
from torch import Tensor
import pickle

def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-exact", 
        action="store_true",
        help="Use exact matching standards for snapshot matching"
    )

_A = TypeVar("_A", np.ndarray, Tensor)

def _canonicalize_array(arr: _A) -> np.ndarray:
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


class NumpySnapshot:
    """Snapshot testing utility for NumPy arrays using .npz format."""
    
    def __init__(
        self, 
        snapshot_dir: str = "tests/_snapshots",
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get the path to the snapshot file."""
        return self.snapshot_dir / f"{test_name}.npz"
    
    def assert_match(
        self, 
        actual: _A | dict[str, _A], 
        test_name: str, 
        force_update: bool = False,
        rtol: float = 1e-4, 
        atol: float = 1e-2,
    ):
        """
        Assert that the actual array(s) matches the snapshot.
        
        Args:
            actual: Single NumPy array or dictionary of named arrays
            test_name: The name of the test (used for the snapshot file)
            update: If True, update the snapshot instead of comparing
        """
        snapshot_path = self._get_snapshot_path(test_name)
        
        # Convert single array to dictionary for consistent handling
        arrays_dict = actual if isinstance(actual, dict) else {"array": actual}
        arrays_dict = {
            k: _canonicalize_array(v)
            for k, v in arrays_dict.items()
        }
        
        
        # Load the snapshot
        expected_arrays = dict(np.load(snapshot_path))
        
        # Verify all expected arrays are present
        missing_keys = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing_keys:
            raise AssertionError(f"Keys {missing_keys} not found in snapshot for {test_name}")
        
        # Verify all actual arrays are expected
        extra_keys = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra_keys:
            raise AssertionError(f"Snapshot contains extra keys {extra_keys} for {test_name}")
        
        # Compare all arrays
        for key in arrays_dict:
            np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key], 
                rtol=rtol, 
                atol=atol,
                err_msg=f"Array '{key}' does not match snapshot for {test_name}"
            )


class Snapshot:
    def __init__(self, snapshot_dir: str = "tests/_snapshots"):
        """
        Snapshot for arbitrary data types, saved as pickle files.
        """
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.pkl"
    
    def assert_match(
        self,
        actual: _A | dict[str, _A],
        test_name: str,
        force_update: bool = False,
    ):
        """
        Assert that the actual data matches the snapshot.
        Args:
            actual: Single object or dictionary of named objects
            test_name: The name of the test (used for the snapshot file)
            force_update: If True, update the snapshot instead of comparing
        """
    
        snapshot_path = self._get_snapshot_path(test_name)


        # Load the snapshot
        with open(snapshot_path, "rb") as f:
            expected_data = pickle.load(f)
        
        if isinstance(actual, dict):
            for key in actual: 
                if key not in expected_data:
                    raise AssertionError(f"Key '{key}' not found in snapshot for {test_name}")
                assert actual[key] == expected_data[key], f"Data for key '{key}' does not match snapshot for {test_name}"
        else:
            assert actual == expected_data, f"Data does not match snapshot for {test_name}"
        

@pytest.fixture
def snapshot(request):
    """
    Fixture providing snapshot testing functionality.
    
    Usage:
        def test_my_function(snapshot):
            result = my_function()
            snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    # Create the snapshot handler with default settings
    snapshot_handler = Snapshot()
    
    # Patch the assert_match method to include the update flag by default
    original_assert_match = snapshot_handler.assert_match
    
    def patched_assert_match(actual, test_name=None, force_update=force_update):
        # If test_name is not provided, use the test function name
        if test_name is None:
            test_name = request.node.name
        return original_assert_match(actual, test_name=test_name, force_update=force_update)
    
    snapshot_handler.assert_match = patched_assert_match
    
    return snapshot_handler



# Fixture that can be used in all tests
@pytest.fixture
def numpy_snapshot(request):
    """
    Fixture providing numpy snapshot testing functionality.
    
    Usage:
        def test_my_function(numpy_snapshot):
            result = my_function()
            numpy_snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    match_exact = request.config.getoption("--snapshot-exact", default=False)
    
    # Create the snapshot handler with default settings
    snapshot = NumpySnapshot()
    
    # Patch the assert_match method to include the update flag by default
    original_assert_match = snapshot.assert_match
    
    def patched_assert_match(actual, test_name=None, force_update=force_update, rtol=1e-4, atol=1e-2):
        # If test_name is not provided, use the test function name
        if test_name is None:
            test_name = request.node.name
        if match_exact:
            rtol = atol = 0
        return original_assert_match(actual, test_name=test_name, force_update=force_update, rtol=rtol, atol=atol)
    
    snapshot.assert_match = patched_assert_match
    
    return snapshot