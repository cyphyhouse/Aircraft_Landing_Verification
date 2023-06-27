import numpy as np 
import matplotlib.pyplot as plt 
import os 
import copy

script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, 'data/data4.txt')
label_path = os.path.join(script_dir, 'estimation_label/label4.txt')
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

# data_path = os.path.join(script_dir, 'data/data2_normalized.npy')
# label_path = os.path.join(script_dir, 'estimation_label/label2_normalized.npy')
# data = np.load(data_path)
# label = np.load(label_path)

# table = [row.strip().split('\n') for row in x]

fraction = 0.99

data_train = data[:,1:]
label_train = label[:,1:]

data_total = data[:90000, 1:2]
ref_total = data[:90000, 1:2]
label_total = label[:90000, 1:2]

data_train = copy.deepcopy(data_total) 
ref_train = copy.deepcopy(ref_total) 
label_train = copy.deepcopy(label_total)

dist_array = np.abs(ref_total-label_total).squeeze()
sorted_dist_idx_array = np.argsort(dist_array)
reduced_array = sorted_dist_idx_array[:int(sorted_dist_idx_array.size*fraction)]
reduced_array = np.sort(reduced_array)

data_train = copy.deepcopy(data_total[reduced_array,:])
ref_train = copy.deepcopy(ref_total[reduced_array,:])
label_train = copy.deepcopy(label_total[reduced_array,:])

plt.plot(ref_train, label_train, 'b*')

plt.show()