# FOR ACTIVATION FUNCTIONS


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 
l1_001= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l1_0.001.csv")
l1_005= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l1_0.005.csv")
l1_01= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l1_0.01.csv")
l1_05= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l1_0.05.csv")
l2_001= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l2_0.001.csv")
l2_005= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l2_0.005.csv")
l2_01= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l2_0.01.csv")
l2_05= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l2_0.05.csv")
l1_l2_001= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_l1_l2_0.001.csv")
relu= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu.csv")
np_l1_001= np.loadtxt(fname = l1_001, delimiter = ',',dtype='double')
np_l1_005= np.loadtxt(fname = l1_005, delimiter = ',',dtype='double')
np_l1_01= np.loadtxt(fname = l1_01, delimiter = ',',dtype='double')
np_l1_05= np.loadtxt(fname = l1_05, delimiter = ',',dtype='double')
np_l2_001= np.loadtxt(fname = l2_001, delimiter = ',',dtype='double')
np_l2_005= np.loadtxt(fname = l2_005, delimiter = ',',dtype='double')
np_l2_01= np.loadtxt(fname = l2_01, delimiter = ',',dtype='double')
np_l2_05= np.loadtxt(fname = l2_05, delimiter = ',',dtype='double')
np_l1_l2_001= np.loadtxt(fname = l1_l2_001, delimiter = ',',dtype='double')
normal_relu= np.loadtxt(fname = relu, delimiter = ',',dtype='double')
np_range= np.arange(25,5025,step=25)

plt.figure(figsize=(8, 6))
plt.plot(np_range,np_l1_001,
        label='L1 0.001',
        linewidth=2)
plt.plot(np_range,np_l1_005,
        label='L1 0.005',
        linewidth=2)
plt.plot(np_range,normal_relu,
        label='NONE ',
        linewidth=2)
plt.plot(np_range,np_l2_001,
        label='L2 0.001',
        linewidth=2)
plt.plot(np_range,np_l2_005,
        label='L2 0.005',
        linewidth=2)
plt.plot(np_range,np_l1_l2_001,
        label='L1 L2 0.001 ',
        linewidth=2)   

"""plt.plot(np_range,np_l2_01,
        label='L2 0.010',
        linewidth=2)
plt.plot(np_range,np_l2_05,
        label='L2 0.050',
        linewidth=2)"""  

# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1600.0])
plt.ylim([0.0, 1.00])
plt.xlabel('Iterations')
plt.ylabel('AUC Scores')
plt.title('Varying Regularization Parameters')
plt.legend(loc="lower right")
plt.show()
