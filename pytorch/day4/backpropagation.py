import numpy as np
import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=torch.tensor([1.0])#初始化
w.requires_grad=True#要对w求梯度，自动求导

def forward(x):
    return w*x#前向传播函数，求预测值
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2#损失函数

for epoch in range(100):
    for x,y in zip(x_data,y_data):#随机梯度下降，每次看一个样本，性能精度好，但时间长#zip把列表数据一对一对绑定
        l=loss(x,y)
        l.backward()#自动算loss对w的偏导
        print('\tgrad:',x,y,w.grad.item())#item取数值
        w.data=w.data-0.01*w.grad.data#改w时只改数值，不改梯度，所以data

        w.grad.data.zero_()#梯度清0
    print("process:",epoch,l.item())
print("predict (after training):",4,forward(4).item())
