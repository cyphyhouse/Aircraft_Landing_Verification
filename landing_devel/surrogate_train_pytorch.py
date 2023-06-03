import torch
import numpy as np 

class SurrogateModel(torch.nn.Module):

    def __init__(self):
        super(SurrogateModel, self).__init__()

        self.linear1 = torch.nn.Linear(6, 32)
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

data_path = 'D:/1_study/1_research/Aircraft_Landing_Verification/landing_devel/data/data.txt'
label_path = 'D:/1_study/1_research/Aircraft_Landing_Verification/landing_devel/estimation_label/label.txt'
data = np.loadtxt(data_path, delimiter=',')
label = np.loadtxt(label_path, delimiter=',')

data_train = data[:8000, 1:]
label_train = label[:8000, 1:]
data_test = data[8000:, 1:]
label_test = label[8000:, 1:]

model = SurrogateModel()

data_train_tensor = torch.FloatTensor(data_train)
label_train_tensor = torch.FloatTensor(label_train)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)
loss = torch.nn.MSELoss()

for i in range(10000):
    predict = model(data_train_tensor)
    error = loss(predict, label_train_tensor)
    print(i, error)
    optimizer.zero_grad()
    error.backward()
    optimizer.step()
    scheduler.step()

torch.save(model, './pytorch_model.pth')    