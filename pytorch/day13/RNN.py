import torch
input_size=4
hidden_size=4
batch_size=1
idx2char=['e','h','l','o']
x_data=[1,0,2,2,3]
y_data=[3,1,2,3,2]

one_hot_lookup=[[1,0,0,0],#e
                [0,1,0,0],#h      
                [0,0,1,0],#l
                [0,0,0,1]]#o 
x_one_hot=[one_hot_lookup[x] for x in x_data]
inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)#把输入变成三维张量，-1 代表自动推断，batch_size=1，input_size=4
labels=torch.LongTensor(y_data).view(-1,batch_size)#把标签变成二维张量，-1 代表自动推断，batch_size=1
class Net(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Net,self).__init__()
        
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.input_size=input_size
        self.rnncell=torch.nn.RNNCell(input_size=self.input_size,hidden_size=self.hidden_size)#RNN 层，输入大小 input_size，隐藏层大小 hidden_size，batch_first=True 表示输入和输出的 batch 大小在第一维
    def forward(self,input,hidden):
        hidden=self.rnncell(input,hidden)#RNNCell 的输入是当前输入和上一个隐藏状态，输出是当前隐藏状态
        return hidden
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)#初始化隐藏状态，大小为 batch_size x hidden_size，初始值为 0
model=Net(input_size,hidden_size,batch_size)
criterion=torch.nn.CrossEntropyLoss()#交叉熵损失函数，-ylog y_pred
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)#优化器，负责更新模型权重，让模型越练越准，Adam 是一种自适应学习率优化算法，适合处理稀疏梯度和非平稳目标
for epoch in range(15):
    loss=0
    optimizer.zero_grad()#清空梯度，准备进行反向传播
    hidden=model.init_hidden()#每个 epoch 都要重新初始化隐藏状态，避免不同 epoch 之间的依赖
    for input,label in zip(inputs,labels):
        hidden=model(input,hidden)#把当前输入和上一个隐藏状态传入模型，得到当前隐藏状态
        loss+=criterion(hidden, label)#计算当前隐藏状态和标签之间的损失值，累加到总损失上
        _,idx=hidden.max(dim=1)#取当前隐藏状态中最大的那个值的索引，作为预测结果
        print(idx2char[idx.item()], end='')#把预测结果的索引转换成字符，打印出来，end='' 表示不换行
        
    loss.backward()#反向传播，计算梯度
    optimizer.step()#更新模型权重
    print('Epoch: %d, Loss: %.4f'%(epoch+1,loss.item()/len(inputs)))