from pathlib import Path

from eargait.preprocessing.load_data_helpers import load
from eargait.preprocessing.rotations import align_gravity_and_convert_ear_to_ebf

HERE = Path(__file__).parent


def load_data_of_interval():
    session = load(HERE.joinpath("test_data/subject01"), 50)
    ear_data = align_gravity_and_convert_ear_to_ebf(session)
    ear_data_short = {}
    interval = {"left_sensor": (3058, 3776), "right_sensor": (3055, 3773)}
    for side in ear_data.keys():
        ear_data_short[side] = ear_data[side][interval[side][0] : interval[side][1]]
    return ear_data_short
