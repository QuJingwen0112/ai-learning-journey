import numpy as np
import torch

xy=np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32);
x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear=torch.nn.Linear(8,1)
        self.sigmoid=torch.nn.Sigmoid() ##self.activate=torch.nn.ReLU()
    def forward(self,x):
        x=self.sigmoid(self.linear)##sigmoid更好，避免出现负值
        return x
model=Model()

criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()