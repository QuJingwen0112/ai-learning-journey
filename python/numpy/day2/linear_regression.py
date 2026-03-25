import numpy as np
x=np.array([1,2,3])
y=np.array([3,5,7])#数据集
Ir=0.1#学习率
w,b=0,0#初始化参数
for _ in range(100):#无需参数循环
    y_pred=w*x+b#预估值函数
    loss=np.mean((y-y_pred)**2)#损失函数，对w，b求偏导确定梯度函数
    grad_w=np.mean(-2*x*(y-y_pred))
    grad_b=np.mean(-2*(y-y_pred))
    w-=Ir*grad_w#梯度决定前进方向
    b-=Ir*grad_b
print("w:",w,"b:",b)#线性回归+MSE（预测误差平方的平均值，即loss函数）
