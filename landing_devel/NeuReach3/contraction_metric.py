import torch 
import torch.nn as nn 
import torch.optim as optim 

class Metric(nn.Module):
    def __init__(self, input, dim, Mlb):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, input**2)
        )
        self.input = input 
        self.Mlb = Mlb
        self.template = (torch.eye(self.input)*self.Mlb).cuda()
        
    def forward(self, x):
        tmp = self.model(x)
        tmp = tmp.reshape((x.shape[0], self.input, self.input))
        res = torch.zeros(tmp.shape).cuda()
        for i in range(x.shape[0]):
            tmp2 = tmp[i,:,:]
            tmp3 = torch.matmul(torch.transpose(tmp2, 0, 1), tmp2)
            res[i,:,:] = self.template + tmp3
        return res
    

