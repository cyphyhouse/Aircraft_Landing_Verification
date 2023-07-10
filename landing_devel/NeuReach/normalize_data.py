import numpy as np
from sklearn import preprocessing
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(script_dir, "../data/data2.txt")
label_path = os.path.join(script_dir, "../estimation_label/label2.txt")

data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')
data_train = data[:, 1:-1]
label_train = label[:, 1:]

# print(data_train)

scaler_data = preprocessing.StandardScaler().fit(data_train)
data_train_normalized = scaler_data.transform(data_train)
# print(data_train_normalized)
# scaler_label = preprocessing.StandardScaler().fit(label_train)
label_train_normalized = scaler_data.transform(label_train)

data_train_normalized = np.hstack((data_train_normalized, data[:,-1:]))

np.save(os.path.join(script_dir, "../data/data2_normalized"), data_train_normalized)
np.save(os.path.join(script_dir, "../estimation_label/label2_normalized"), label_train_normalized)
# np.save("/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/NeuReach/data_mean", scaler_data.mean_)
# np.save("/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/NeuReach/data_var", scaler_data.scale_)
# np.save("/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/NeuReach/label_mean", scaler_label.mean_)
# np.save("/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/NeuReach/label_var", scaler_label.scale_)