import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])#图片变成 PyTorch 能用的张量；归一化：把0-255像素压缩到-1~1（让数据更稳定），官方均值和标准差，让数据均值 = 0，让数据方差 = 1

train_data=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transform)
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=0)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)#卷积层，输入 1 个通道（灰度图），输出 10 个通道，卷积核大小 5x5
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.mp=torch.nn.MaxPool2d(2)#最大池化层，窗口大小 2x2，步长 2，作用是把特征图缩小一半，减少计算量
        self.l1=torch.nn.Linear(320,50)#全连接层，输入 320 个特征，输出 50 个特征
        self.l2=torch.nn.Linear(50,10)#全连接层，输入 50 个特征，输出 10 个特征，对应 10 个数字类别     
        

    def forward(self,x):
        batch_size=x.size(0)#获取 batch 大小，-1 代表自动推断
        x=F.relu(self.mp(self.conv1(x)))#卷积 -> 激活函数 -> 池化
        x=F.relu(self.mp(self.conv2(x)))#卷积 -> 激活函数 -> 池化
        x=x.view(batch_size,-1)#把特征图展平成一行向量  
        x=self.l1(x)
        x=F.relu(x)
        x=self.l2(x)
        return x
model=Net()
criterion=torch.nn.CrossEntropyLoss()#交叉熵损失函数，-ylog y_pred
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)#优化器，负责更新模型权重，让模型越练越准，momentum 冲量，加速训练
def train(epoch):
    running_loss=0.0#用来累计损失值，每 300 个批次打印一次平均 loss
    for batch_idx,data in enumerate(train_loader,0):
        inputs,targets=data
        optimizer.zero_grad()
        output=model(inputs)
        loss=criterion(output,targets)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()#把每一批的损失值加起来
        if batch_idx%300==299:#每走 300 个批次 打印一次损失
            print('[%d,%5d] loss: %.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0#打印完清空损失，重新累计下一组 300 批次
def test():
    correct=0#记录预测正确的数量
    total=0#记录总共有多少张图片
    with torch.no_grad():#测试时不计算梯度，节省内存、加速计算
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)#输出 10 个分数，取最大的那个作为预测数字，_ = 我不需要这个值，只是占个位置，我们只需要位置
            total+=labels.size(0)#累计总图片数量（一个 batch64 张）
            correct+=(predicted==labels).sum().item()#
    print('Accuracy on the 10000 test images: %d %%'%(100*correct/total))
if __name__=='__main__':#表示下面的代码只有直接运行时才执行
    for epoch in range(10):
        train(epoch)
        test()  

        
