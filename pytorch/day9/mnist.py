import torch
from  torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transforms.ToTensor())

test_data=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transforms.ToTensor())

train_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=0)

test_loader=DataLoader(dataset=test_data,batch_size=32,shuffle=False,num_workers=0)

for images,labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break   