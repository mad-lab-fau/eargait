import pandas as pd
import pprint
from pathlib import Path

from eargait.typed import GaitSequenceAnalyzerPipeline
from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf, load
from eargait.utils import TrimMeanGravityAlignment
from eargait.utils.example_data import get_mat_example_data_path

BASE_PATH = Path(r"C:\Users\jahne\Desktop\HIWI\control_data")


GROUP = "healthy"
SUBJECT = "ac74"
SPEED = "fast"

'''DATA_PATHS = {
    "healthy": {
        "subjects": ["ac74", "as13", "el80", "es83", "fs63", "jw53", "kh43",
                     "mm45", "mq17", "ns20", "op86", "os31", "ov11", "pb22", "ql19",
                     "qu66", "rr77", "sc27", "si62", "sw33", "ve95", "vw55", "yg51", "zu72"]
    },
    "geriatry": {
        "subjects": ["cd90", "dp99", "er71", "fa49", "fr44", "hh39", "hi15", "kp29", "ks72",
                     "lo42", "me58", "ow24", "pa41", "po60", "qc87", "ru93", "sv18", "wt21",
                     "wu78", "xy82", "zi36"]
    }
}'''

# Automatically build path:
def build_data_path(group: str, subject: str, speed: str):
    return BASE_PATH / group / "data" / subject / "walking" / "signia" / speed

# Final data path:
data_path = build_data_path(GROUP, SUBJECT, SPEED)


#data_path = get_mat_example_data_path()
target_sample_rate = 50
session = load(data_path, target_sample_rate_hz=target_sample_rate, skip_calibration=True)
trim_method = TrimMeanGravityAlignment(sampling_rate_hz=target_sample_rate)
ear_data = align_gravity_and_convert_ear_to_ebf(session, trim_method)
ear_data = ear_data['left_sensor']
#ear_data = ear_data
manual_sequences = pd.DataFrame(
    {
        "start": [1050, 1800, 2700, 3750, 4650, 5550, 6450, 7350],
        "end": [1500, 2250, 3300, 4350, 5250, 6150, 7050, 7800],
    }
)
'''manual_sequences = pd.DataFrame(
    {
        "start": [1850, 1400],
        "end": [1500, 2250],
    }
)'''


combi2 = GaitSequenceAnalyzerPipeline(sample_rate=50, strictness=0, min_seq_length=1)
combi2.compute(ear_data)
#combi2.compute_gait_features_manual(ear_data, manual_sequences)

pprint.pprint(("Parameter automatic", combi2.auto_average_params_))
pprint.pprint(("List automatic", combi2.auto_sequence_list_))

#pprint.pprint(("Parameter manual", combi2.manual_average_params_))
#pprint.pprint(("List manual", combi2.manual_sequence_list_))


"""
Gestestete Dinge:
Manual:
    - Keine Sequences vorhanden --> Value Error if empty with suggestions hinzugefügt in run eargait 
    - Same length --> already implemented
    - Wrong start oder end name --> already implemented 
    - Overlapping manual sequences --> Warning
    - Müssen wir out of bound, start < end > start checken oder erwarten wir das vom user? Ja abfragen! value error 
    - sequence to short --> already implemented (greater than 3 sec)
Auto:
    - Wenn input kein df mit links oder rechts ist --> type check df or dict --> noch genauer checken? 

allgemein:
    - falsche sample rate --> already implemented 
    - müssen wir session sample rate vs Pipeline sample rate checken? --> macht sinn ja session vs input zu pipeline geht nicht wegen nicht jeder hat session 
    - WICHTIG - Min seq length 1 und dann exakt -> Diao wirft fehler --> wie lösen? - checken mit anderer pip ist es da nicht aufgefallen 
"""




# TODO liste:
    # - sequence to small error beheben / Diao change >= change
        # Nicht aufgefallen in previous pipeline, da wir min seq length auf 2 gesetzt hatten (Notizen) -> Grund? idk
        # wenn min seq length 1 -> 150 min samples -> diao_event detection -> _filter data reduziert um delay x -> geht um ca 1/3 (bissl weniger 45 bei 50hz ; ~180 bei 200hz)
        # dadurch unter >= 3* sample rate threshold ... Was tun
        # TODO AKS fragen 
    # - Diao and random forest testen mit anderen algos (CNN sztatt RF; event detection anderer)
    # - Direkt am anfang richtiges format abfangen!! vor irgendwelchen Berechnungen in compute
    # - Mehr probanden testen - randcases überlegen
    # - Neues format mit local files
    # - Activity als required argument für pipeline with default als walking für pipeline