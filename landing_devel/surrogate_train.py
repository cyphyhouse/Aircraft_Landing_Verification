import torch
import numpy as np 
import os 
import copy

class SurrogateModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(SurrogateModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 32)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 32)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32, 32)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(32, 6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        return x

def loss(predict, label):
    error = 1/predict.shape[0]*torch.sum(
        (predict[:,0]-label[:,0])**2 + \
        (predict[:,1]-label[:,1])**2 + \
        (predict[:,2]-label[:,2])**2 + \
        1000000*(predict[:,3]-label[:,3])**2 + \
        1000000*(predict[:,4]-label[:,4])**2 + \
        1000000*(predict[:,5]-label[:,5])**2 
    )
    return error 

script_dir = os.path.realpath(os.path.dirname(__file__))

data_path = os.path.join(script_dir, 'data/data.txt')
label_path = os.path.join(script_dir, 'estimation_label/label.txt')

data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

data_train = data[:45000, 1:]
label_train = label[:45000, 1:]
data_test = data[45000:, 1:]
label_test = label[45000:, 1:]

input_dim = data_train.shape[1]

model = SurrogateModel(input_dim)

data_train_tensor = torch.FloatTensor(data_train)
label_train_tensor = torch.FloatTensor(label_train)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)
# loss = torch.nn.MSELoss()

best_loss = float('inf')
best_model = None 

for i in range(100000):
    predict = model(data_train_tensor)
    error = loss(predict, label_train_tensor)
    if error < best_loss:
        print(i, error)
        best_loss = error 
        best_model = copy.deepcopy(model)
    optimizer.zero_grad()
    error.backward()
    optimizer.step()
    scheduler.step()

torch.save(best_model.state_dict(), os.path.join(script_dir,'./surrogate_model_06-05_lighting.pth'))    