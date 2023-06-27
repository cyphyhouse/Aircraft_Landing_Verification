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
window_width = 100

data_train = data[:,1:]
label_train = label[:,1:]

data_total = data[:90000, 1:2]
ref_total = data[:90000, 1:2]
label_total = label[:90000, 1:2]

data_length = data_total.shape[0]
data_max = np.max(data_total[:,0])
data_min = np.min(data_total[:,0])
bin_array = np.arange(data_min, data_max, window_width)
data_dict = {}
ref_dict = {}
idx_dict = {}
label_dict = {}
for i in range(data_length):
    for bin_lb in bin_array:
        if data_total[i,0] > bin_lb and data_total[i,0] < bin_lb + window_width:
            if bin_lb not in data_dict:
                data_dict[bin_lb] = [data_total[i,0]]
                ref_dict[bin_lb] = [ref_total[i,0]]
                idx_dict[bin_lb] = [i]
                label_dict[bin_lb] = [label_total[i,0]]
            else:
                data_dict[bin_lb].append(data_total[i,0])
                ref_dict[bin_lb].append(ref_total[i,0])
                idx_dict[bin_lb].append(i)
                label_dict[bin_lb].append(label_total[i,0])
kept_idx = []
for key in data_dict:
    dist_array = np.abs(np.array(ref_dict[key])-np.array(label_dict[key]))
    sorted_dist_idx_array = np.argsort(dist_array)
    reduced_array = sorted_dist_idx_array[:round(sorted_dist_idx_array.size*fraction)]
    reduced_array = np.sort(reduced_array)
    kept_idx += (np.array(idx_dict[key])[reduced_array]).tolist()

data_train = copy.deepcopy(data_total[kept_idx,:])
ref_train = copy.deepcopy(ref_total[kept_idx,:])
label_train = copy.deepcopy(label_total[kept_idx,:])



plt.plot(ref_train, label_train, 'b*')

plt.show()