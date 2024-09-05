"""Copy pasted from eargait. ?? jetzt nicht mehr in eargait?"""
import numba
import numpy as np


def _solve_overlap(input_array: np.ndarray, gap_size: int) -> numba.typed.List:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other."""
    stack = numba.typed.List()
    stack.append(input_array[0])

    for i in range(1, len(input_array)):
        if stack[-1][0] <= input_array[i][0] <= (stack[-1][1] + gap_size) <= (input_array[i][1] + gap_size):
            stack[-1][1] = input_array[i][1]
        else:
            stack.append(input_array[i])

    return stack


def merge_intervals(input_array: np.ndarray, gap_size: int = 0) -> np.ndarray:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other.

    This is actually a wrapper for _solve_overlap that is needed because numba can not compile np.sort().

    Parameters
    ----------
    input_array : (n, 2) np.ndarray
        The np.ndarray containing the intervals that should be merged
    gap_size : int
        Integer that sets the allowed gap between intervals.
        For examples see below.
        Default is 0.

    Returns
    -------
    merged intervals array
        (n, 2) np.ndarray containing the merged intervals

    Examples
    --------
    >>> tests = np.array([[1, 3], [2, 4], [6, 8], [5, 7], [10, 12], [11, 15], [18, 20]])
    >>> merge_intervals(tests)
    array([[ 1,  4],
           [ 5,  8],
           [10, 15],
           [18, 20]])

    >>> merge_intervals(tests, 2)
    array([[ 1, 15],
           [18, 20]])

    """
    return np.array(_solve_overlap(np.sort(input_array, axis=0, kind="stable"), gap_size))
