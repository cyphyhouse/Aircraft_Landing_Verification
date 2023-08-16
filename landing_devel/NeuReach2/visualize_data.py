import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, 'data.pickle')
with open(data_file_path,'rb') as f:
    data = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x0 = []
e = []
for i in range(len(data)):
    traces = data[i][1]
    tmp = np.reshape(traces,(-1,6))
    tmp = np.mean(tmp, axis=0)
    # if np.linalg.norm(tmp)>1e5:
    #     continue
    e.append(tmp)
    X0 = data[i][0]
    x, Ec, Er = X0 
    x0.append(x)

x0 = np.array(x0)
e = np.array(e)
print(np.max(np.abs(x0[:,0])),np.max(np.abs(x0[:,1])),np.max(np.abs(x0[:,2])))
ax.scatter(x0[:,0], x0[:,1], x0[:,2])
# ax.scatter(e[:,0], e[:,1], e[:,2])
plt.show()