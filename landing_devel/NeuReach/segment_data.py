import numpy as np 

import os 

script_dir = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(script_dir, '../data/data3.txt')
label_path = os.path.join(script_dir, '../estimation_label/label3.txt')

data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

# data = data[:,:]
# label = label[:,1:]

data_0 = []
label_0 = []
data_1 = []
label_1 = []
data_2 = []
label_2 = []
data_3 = []
label_3 = []
data_4 = []
label_4 = []
data_5 = []
label_5 = []
data_6 = []
label_6 = []

for i in range(data.shape[0]):
    if data[i,-1] >=0.55 and data[i,-1] < 0.65:
        data_0.append(data[i,:])
        label_0.append(label[i,:])
    elif data[i,-1] >=0.65 and data[i,-1] < 0.75:
        data_1.append(data[i,:])
        label_1.append(label[i,:])
    elif data[i,-1] >=0.75 and data[i,-1] < 0.85:
        data_2.append(data[i,:])
        label_2.append(label[i,:])
    elif data[i,-1] >=0.85 and data[i,-1] < 0.95:
        data_3.append(data[i,:])
        label_3.append(label[i,:])
    elif data[i,-1] >=0.95 and data[i,-1] < 1.05:
        data_4.append(data[i,:])
        label_4.append(label[i,:])
    elif data[i,-1] >=1.05 and data[i,-1] < 1.15:
        data_5.append(data[i,:])
        label_5.append(label[i,:])
    elif data[i,-1] >=1.15 and data[i,-1] <= 1.25:
        data_6.append(data[i,:])
        label_6.append(label[i,:])
    else:
        pass 

data_0 = np.array(data_0)
label_0 = np.array(label_0)
data_1 = np.array(data_1)
label_1 = np.array(label_1)
data_2 = np.array(data_2)
label_2 = np.array(label_2)
data_3 = np.array(data_3)
label_3 = np.array(label_3)
data_4 = np.array(data_4)
label_4 = np.array(label_4)
data_5 = np.array(data_5)
label_5 = np.array(label_5)
data_6 = np.array(data_6)
label_6 = np.array(label_6)

np.savetxt(os.path.join(script_dir,'../data/data3_055-065.txt'),data_0,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_055-065.txt'),label_0,delimiter=',')
np.savetxt(os.path.join(script_dir,'../data/data3_065-075.txt'),data_1,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_065-075.txt'),label_1,delimiter=',')
np.savetxt(os.path.join(script_dir,'../data/data3_075-085.txt'),data_2,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_075-085.txt'),label_2,delimiter=',')
np.savetxt(os.path.join(script_dir,'../data/data3_085-095.txt'),data_3,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_085-095.txt'),label_3,delimiter=',')
np.savetxt(os.path.join(script_dir,'../data/data3_095-105.txt'),data_4,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_095-105.txt'),label_4,delimiter=',')
np.savetxt(os.path.join(script_dir,'../data/data3_105-115.txt'),data_5,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_105-115.txt'),label_5,delimiter=',')
np.savetxt(os.path.join(script_dir,'../data/data3_115-125.txt'),data_6,delimiter=',')
np.savetxt(os.path.join(script_dir,'../estimation_label/label3_115-125.txt'),label_6,delimiter=',')
