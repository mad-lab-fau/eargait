"""Sets globally used variables, constant over whole project."""
LABELS = [
    "jogging",
    "biking",
    "walking",
    "sitting",
    "lying",
    "jumping",
    "stairs up",
    "stairs down",
    "stand",
    "transition",
]
LABELS_TO_INT = {
    "jogging": 0,
    "biking": 1,
    "walking": 2,
    "sitting": 3,
    "lying": 4,
    "jumping": 5,
    "stairs up": 6,
    "stairs down": 7,
    "stand": 8,
    "transition": 9,
}
INT_TO_LABELS = {
    0: "jogging",
    1: "biking",
    2: "walking",
    3: "sitting",
    4: "lying",
    5: "jumping",
    6: "stairs up",
    7: "stairs down",
    8: "stand",
    9: "transition",
}
TRANSITION_LABELS = ["lowering", "rising", "bending", "flopping"]
EXCLUDED_LABELS = ["fall_no_recover", "fall_recover", "near_fall", "nodding"]
GYRO_COLUMNS = ["hiGyrX", "hiGyrY", "hiGyrZ"]
ACC_COLUMNS = ["x", "y", "z"]
ALL_COLUMNS = [*ACC_COLUMNS, *GYRO_COLUMNS]
PARTICIPANTS = 22
TRANSITION_THRESHOLD = 0.5
PERIODIC_THRESHOLD = 0.8
DEFECT_FILES = ["ID18_AB7_Activity_Labels.csv", "ID04_AB6_Activity_Labels.csv"]
NUM_WORKERS = 0  # workers for dataloader, 4 is max recommended of .hpc (before 4 > might cause DataLoader Error)
ML_LOGGING_FOLDER = "ml_logging/"
DL_LOGGING_FOLDER = "lightning_logs"
DL_MODELS = ["base_cnn", "conv_gru"]
ML_MODELS = ["svm", "knn", "adaboost", "random_forest"]
SEED = 42
