"""
    混淆矩阵图画法：CK+  ENTERFACE05   BRED

"""

# CK+
# from __future__ import division
# from numpy import *
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
#
# def plotCM(classes, matrix, savname):
#
#     matrix = matrix.astype(np.float)
#     linesum = matrix.sum(1)
#     linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
#     matrix /= linesum
#     matrix*=100
#     matrix = np.round(matrix, 2)  # 小数点位数
#     font1 = {
#              'color': 'white',
#              }
#
#     fig = plt.figure(figsize=(7, 7))
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(matrix, cmap=plt.cm.get_cmap('gray_r'))
#     #fig.colorbar(cax)
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.yaxis.set_major_locator(MultipleLocator(1))
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             if(i==j):
#                 ax.text(i, j, str(matrix[j][i]), va='center', ha='center', fontdict = font1)
#             else:
#                 ax.text(i, j, str(matrix[j][i]), va='center', ha='center')
#
#     ax.set_xticklabels([''] + classes, rotation=45)
#     ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
#     ax.set_yticklabels([''] + classes)
#
#     plt.savefig(savname,dpi=400)
#
# classes=["An","Co","Di","Fe","Ha","Sa","Su"]
#
# matrix=np.zeros([7,7])
# matrix[0][0]=10000
# matrix[1][1]=9600
# matrix[1][4]=400
# matrix[2][0]=140
# matrix[2][2]=9860
# matrix[3][3]=10000
# matrix[4][1]=250
# matrix[4][4]=9750
# matrix[5][1]=1000
# matrix[5][3]=500
# matrix[5][5]=8500
# matrix[6][5]=170
# matrix[6][6]=9830
#
# plotCM(classes,matrix,'img.jpg')

#enterface
# from __future__ import division
# from numpy import *
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
#
# def plotCM(classes, matrix, savname):
#
#     matrix = matrix.astype(np.float)
#     linesum = matrix.sum(1)
#     linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
#     matrix /= linesum
#     matrix*=100
#     matrix = np.round(matrix, 2)  # 小数点位数
#     font1 = {
#              'color': 'white',
#              }
#
#     fig = plt.figure(figsize=(7, 7))
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(matrix, cmap=plt.cm.get_cmap('gray_r'))
#     #fig.colorbar(cax)
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.yaxis.set_major_locator(MultipleLocator(1))
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             if(i==j):
#                 ax.text(i, j, str(matrix[j][i]), va='center', ha='center', fontdict = font1)
#             else:
#                 ax.text(i, j, str(matrix[j][i]), va='center', ha='center')
#
#     ax.set_xticklabels([''] + classes, rotation=45)
#     ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
#     ax.set_yticklabels([''] + classes)
#
#     plt.savefig(savname,dpi=400)
#
# classes=["An","Di","Fe","Ha","Sa","Su"]
#
# matrix=np.zeros([6,6])
# matrix[0][0]=48
# matrix[0][1]=16
# matrix[0][2]=16
# matrix[0][3]=2
# matrix[0][5]=18
# matrix[1][0]=6
# matrix[1][1]=88
# matrix[1][2]=2
# matrix[1][5]=4
# matrix[2][0]=4
# matrix[2][1]=4
# matrix[2][2]=72
# matrix[2][5]=20
# matrix[3][0]=4
# matrix[3][3]=90
# matrix[3][5]=6
# matrix[4][1]=22
# matrix[4][2]=30
# matrix[4][3]=2
# matrix[4][4]=32
# matrix[4][5]=14
# matrix[5][2]=34
# matrix[5][5]=66
#
#
# plotCM(classes,matrix,'img.jpg')
#


#BRED
from __future__ import division
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plotCM(classes, matrix, savname):

    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    matrix*=100
    matrix = np.round(matrix, 2)  # 小数点位数
    font1 = {
             'color': 'white',
             }

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.cm.get_cmap('gray_r'))
    #fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(i==j):
                ax.text(i, j, str(matrix[j][i]), va='center', ha='center', fontdict = font1)
            else:
                ax.text(i, j, str(matrix[j][i]), va='center', ha='center')

    ax.set_xticklabels([''] + classes, rotation=45)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_yticklabels([''] + classes)

    plt.savefig(savname,dpi=400)

classes=["An","Di","Fe","Ha","Sa","Su"]


pre = np.append(np.load("experiment/matrxi/pre1_0.93.npy"),np.load("experiment/matrxi/pre1_0.92.npy"))
# pre1 = np.append(pre,np.load("experiment/matrxi/pre1_0.93.npy"))
#np.load("experiment/matrxi/pre1_0.92.npy")+np.load("experiment/matrxi/pre1_0.94.npy")+np.load("experiment/matrxi/pre1_0.978.npy")
pre = np.append(pre,np.load("experiment/matrxi/pre1_0.94.npy"))
pre = np.append(pre,np.load("experiment/matrxi/pre1_0.978.npy"))
target = np.append(np.load('experiment/matrxi/target1_0.978.npy'),np.load('experiment/matrxi/target1_0.978.npy'))
target = np.append(target,target)
corrects = float(sum(pre == target)) / float(target.size)
print(corrects)

from sklearn import metrics
matrix = metrics.confusion_matrix(target, pre)
classes = ["An", "Di", "Fe", "Ha", "Sa", 'Su']
plotCM(classes, matrix, 'Code/wj.jpg')

matrix=np.zeros([6,6])
matrix[0][0]=4166
matrix[0][1]=3799
matrix[0][2]=2036


matrix[1][0]=3095
matrix[1][1]=3125
matrix[1][4]=2505
matrix[1][5]=1275


matrix[2][1]=830
matrix[2][2]=6250
matrix[2][4]=2920

matrix[3][0]=569
matrix[3][2]=924
matrix[3][3]=5938
matrix[3][5]=2569

matrix[4][1]=2121
matrix[4][2]=1941
matrix[4][4]=5938

matrix[5][2]=1250
matrix[5][3]=4167
matrix[5][5]=4583


plotCM(classes,matrix,'img.jpg')

