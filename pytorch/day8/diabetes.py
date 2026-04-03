import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from sklearn.datasets import load_diabetes

class DiabetesDataset(Dataset):
    def __init__(self):
        diabetes=load_diabetes()
        x=diabetes.data
        y=(diabetes.target>np.median(diabetes.target)).astype(np.float32)#把数据集原始健康程度标签求中位数，用以判断是否患病，把True/False成32位浮点数
        y=y.reshape(-1,1)#变成 [样本数，1] 的二维矩阵

        self.len=x.shape[0]#获取样本总数
        self.x_data=torch.from_numpy(x.astype(np.float32))
        self.y_data=torch.from_numpy(y)#把 numpy 数组 → 转换成 PyTorch 张量（Tensor）

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(10,1)
        self.sigmoid=torch.nn.Sigmoid()#定义 Sigmoid 激活函数，把输出压缩到 0~1 之间，用于二分类概率
    def forward(self,x):
        x=self.sigmoid(self.linear(x))#线性计算 → 经过 Sigmoid → 输出 0~1 之间的概率
        return x
    
dataset=DiabetesDataset()
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=0)##dataset：传入我们的数据集：一次喂 32 个样本：每个 epoch 打乱数据：单线程加载

model=Model()
criterion=torch.nn.BCELoss()#二分类交叉熵损失函数，计算预测值和真实标签之间的误差
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)#随机梯度下降优化器，根据损失自动更新权重 w 和 b

loss_list=[]#创建一个空列表，保存每一轮的 loss，用来画图

if __name__=='__main__':#Python 主程序入口，Windows 运行多线程必须写这句
    for epoch in range(100):
        epoch_loss=0.0#每一轮开始前，把累计 loss 清零
        for i,data in enumerate(train_loader,0):#遍历数据加载器，拿出一批 inputs 和 labels，i：批次编号，data：一组数据（输入 + 标签）
            inputs,labels=data

            y_pred=model(inputs)
            loss=criterion(y_pred,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()#把每一批的 loss 累加到 epoch_loss
        avg_loss = epoch_loss / len(train_loader)#
        loss_list.append(avg_loss)#
        print(f'Epoch{epoch+1},Loss={loss.item():.4f}')#打印轮数和当前 loss

    plt.plot(range(1, 101), loss_list)#x 轴：1~100 轮，y 轴：每轮 loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.show()