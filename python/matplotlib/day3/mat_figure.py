import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2

#plt.figure()#第一张图
#plt.plot(x,y1)

plt.figure()#第二张图
plt.plot(x,y2)#*
plt.plot(x,y1,color="red",linewidth=1.0,linestyle="--")#虚线红色，可以改变宽度，默认1.0

plt.xlabel("I'm x")#坐标轴名称
plt.ylabel("I'm y")

plt.plot(x,y1,label='up')#显示图标，注明线条所属，使用时与*二选一，可以定义color,linewidth,linestyle
plt.plot(x,y2,label='dowm')
plt.legend()#可以定义loc='best'等

plt.show()