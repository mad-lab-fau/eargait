import pickle
from signialib import Session

data_path_pkl = r"C:\Users\jahne\Desktop\Eargait\eargait\example_data\pkl_files\data_.pkl"
data_path_mat = r"C:\Users\jahne\Desktop\Eargait\eargait\example_data\mat_files\normal"

with open(data_path_pkl, "rb") as f:
    data_pkl = pickle.load(f)
print(data_pkl["meta_data"]["left_sensor"])
data_pkl["meta_data"]["left_sensor"]["file_name"] = "data_"
data_pkl["meta_data"]["right_sensor"]["file_name"] = "data_"
print(data_pkl["meta_data"]["left_sensor"])


session = Session.from_folder_path(data_path_mat)
print(data_pkl)