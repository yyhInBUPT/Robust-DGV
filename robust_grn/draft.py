import argparse


# import torch
# from torch import Tensor
# import numpy as np
# from utils import load_citation
# import sys
# import pickle as pkl
# dataset_str="citeseer"
# names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
# objects = []
# dataset = dataset_str.title()
# for i in range(len(names)):
#     with open("dataset/" + dataset + "/raw/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
#         if sys.version_info > (3, 0):
#             objects.append(pkl.load(f, encoding='latin1'))
#         else:
#             objects.append(pkl.load(f))
#
# x, y, tx, ty, allx, ally, graph = tuple(objects)
# nodes = tx.shape[0] + allx.shape[0]
# if dataset_str == "citeseer":
#     nodes = nodes + 15
# degree = torch.zeros(nodes)
# for i in range(nodes):
#     degree[i]=len(graph[i])
# print(degree)
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False#显示负号
plt.style.use('seaborn-whitegrid')
x = np.array([1,2,3,4,5,6])
y = np.array([])
A = np.array([0.708,0.676,0.302,0.183,0.250,0.200])
B= np.array([0.682,0.689,0.706,0.729,0.734,0.734])
# C=np.array([0.7965,0.6679,0.5492,0.4405,0.3502,0.2808])
# D=np.array([0.339,0.3363,0.3261,0.3204,0.303,0.3036])

#label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
#color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
#线型：-  --   -.  :    ,
#marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(12,6))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# ax2.grid(linestyle="-")
plt.rcParams.update({'hatch.color':'white', 'hatch.linewidth':'1'})
# ax = plt.gca()
# ax.spines['top'].set_visible(False) #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
# ax = plt.axes()
# ax.set_facecolor("aliceblue")
line1=plt.plot(x,A,color="#699ed4",label="Accuracy of GCN",linewidth=3,linestyle='--',marker='o',markersize=8)
line2=plt.plot(x,B,"#ef8183",label="Accuracy of GCNII",linewidth=3,linestyle='--',marker='o',markersize=8)#sandybrown

bar1 = ax2.bar(np.array([1]),np.array([0.7056]),color="#a4d9bb",hatch='///', label="Robust Ratio of GCN",width=0.4)
bar2 = ax2.bar(np.array([5]),np.array([0.2997]),color="#b88cc0",hatch='///', label="Robust Ratio of GCNII",width=0.4)
# bar2 = ax2.bar(np.array([2]),np.array([0.8000]),ec='skyblue', ls='-', lw=5,color='white',width=0.4)
# plt.plot(x,C,color="red",label="C algorithm",linewidth=1.5)
# plt.plot(x,D,"r--",label="D algorithm",linewidth=1.5)

group_labels=['2','4','8','16','32',' 64'] #x轴刻度的标识
# group_labels=['2','4','6','8','10','12']
plt.xticks(x,group_labels,fontsize=15,) #默认字体大小为10 fontweight='bold'
plt.yticks(fontsize=10,)
# plt.title("example",fontsize=12,fontweight='bold') #默认字体大小为12
# plt.xlabel("Layers",fontsize=20,fontweight='bold')# ,fontweight='bold'
# plt.ylabel("Accuracy",fontsize=20,fontweight='bold')
# plt.xlabel("Perturbation",fontsize=20,fontweight='bold')# ,fontweight='bold'
# plt.ylabel("Robust ratio",fontsize=20,fontweight='bold')
ax1.set_xticks(x,group_labels,fontsize=10,)
# ax1.set_yticks(fontsize=10,fontweight='bold')
ax1.set_xlabel('Layers',fontdict={'size':15,})
ax1.set_ylabel('Accuracy',fontdict={'size': 15,})
ax2.set_ylabel('Robust Ratio',fontdict={'size': 15,})
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
# plt.xlim(3,21) #设置x轴的范围
# plt.ylim(0,1)
# ax1.legend(loc='upper right',fontsize=15)
ax2.legend(loc='upper left',fontsize=10)
#plt.legend()          #显示各曲线的图例
# plt.legend(loc=0, numpoints=1,fontsize=20)
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=20,fontweight='bold')
plt.grid(linestyle = "-") #设置背景网格线为虚线

plt.show()



