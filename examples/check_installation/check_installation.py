r"""

Check installation
=====================

"""

# %%
# Loading a few functions
# -------------------------
#

from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.preprocessing import aling_gravity_and_convert_ear_to_ebf, convert_ear_to_ebf, load
from eargait.utils.example_data import get_example_data_path

# data directory
data_path = get_example_data_path()

print("\nInstallation was successful.\n")
