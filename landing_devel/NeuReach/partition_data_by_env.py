import numpy as np

data_path = "/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data2.txt"
label_path = "/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label2.txt"

data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')
data_train = data[:, 1:]
label_train = label[:, 1:]

data_to_save = []
label_to_save = []
for i in range(len(data_train)):
    if data_train[i, -1] >= 0.95 and data_train[i, -1] < 1.05:
        data_to_save.append(data_train[i, 0:-1])
        label_to_save.append(label_train[i, :])


np.save("/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data2_095_to_105", np.array(data_to_save))
np.save("/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label2_095_to_105", np.array(label_to_save))