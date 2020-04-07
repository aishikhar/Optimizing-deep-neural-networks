# FOR ACTIVATION FUNCTIONS


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 
hardtanh= open("C:\Users\MAHE\Desktop\Programs\Python\micro_hardtanh.csv")
tanh= open("C:\Users\MAHE\Desktop\Programs\Python\micro_tanh.csv")
relu= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu.csv")
sigmoid= open("C:\Users\MAHE\Desktop\Programs\Python\micro_sigmoid.csv")
leakyrelu= open("C:\Users\MAHE\Desktop\Programs\Python\micro_leakyrelu.csv")
elu= open("C:\Users\MAHE\Desktop\Programs\Python\micro_elu.csv")
softplus= open("C:\Users\MAHE\Desktop\Programs\Python\micro_softplus.csv")
np_hardtanh= np.loadtxt(fname = hardtanh, delimiter = ',',dtype='double')
np_relu= np.loadtxt(fname = relu, delimiter = ',',dtype='double')
np_sigmoid= np.loadtxt(fname = sigmoid, delimiter = ',',dtype='double')
np_tanh= np.loadtxt(fname = tanh, delimiter = ',',dtype='double')
np_elu= np.loadtxt(fname = elu, delimiter = ',',dtype='double')
np_softplus= np.loadtxt(fname = softplus, delimiter = ',',dtype='double')
np_leakyrelu= np.loadtxt(fname = leakyrelu, delimiter = ',',dtype='double')
np_range= np.arange(25,5025,step=25)
print '\nrange: ',np_range
plt.figure(figsize=(8, 6))
plt.plot(np_range,np_hardtanh,
        label='HardTanh',
        linewidth=2)
plt.plot(np_range,np_relu,
        label='ReLu',
        linewidth=2)
 
plt.plot(np_range,np_elu,
        label='ELU',
        linewidth=2)

plt.plot(np_range,np_softplus,
        label='SoftPlus',
        linewidth=2)
plt.plot(np_range,np_leakyrelu,
        label='LeakyReLu',
        linewidth=2)        
plt.plot(np_range,np_sigmoid,
        label='Sigmoid',
        linewidth=2)        
plt.plot(np_range,np_tanh,
        label='Tanh',
        linewidth=2)   


# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 2500.0])
plt.ylim([0.0, 1.00])
plt.xlabel('Iterations ')
plt.ylabel('AUC Score (micro)')
plt.title('AUC Scores for different Activation Functions')
plt.legend(loc="lower right")
plt.show()
