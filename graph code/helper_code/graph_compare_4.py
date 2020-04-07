# FOR VARYING UPDATERS


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 
nesterov= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_nesterov.csv")
adam= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_adam.csv")
adadelta= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_adadelta.csv")
rmsprop= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_rmsprop.csv")
sgd= open("C:\Users\MAHE\Desktop\Programs\Python\micro_relu_sgd_sgd.csv")

np_nesterov= np.loadtxt(fname = nesterov, delimiter = ',',dtype='double')
np_adam= np.loadtxt(fname = adam, delimiter = ',',dtype='double')
np_adadelta= np.loadtxt(fname = adadelta, delimiter = ',',dtype='double')
np_rmsprop= np.loadtxt(fname = rmsprop, delimiter = ',',dtype='double')
np_sgd= np.loadtxt(fname = sgd, delimiter = ',',dtype='double')
np_range= np.arange(25,5025,step=25)

plt.figure(figsize=(8, 6))
plt.plot(np_range,np_nesterov,
        label='Momentum Nesterov',
        linewidth=2)
plt.plot(np_range,np_adam,
        label='Adam',
        linewidth=2)
plt.plot(np_range,np_rmsprop,
        label='RMSProp',
        linewidth=2)
plt.plot(np_range,np_adadelta,
        label='AdaDelta',
        linewidth=2)
plt.plot(np_range,np_sgd,
        label='SGD',
        linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1600.0])
plt.ylim([0.0, 1.00])
plt.xlabel('Iterations')
plt.ylabel('AUC Scores')
plt.title('Varying Updation Algorithms for Learning Rate')
plt.legend(loc="lower right")
plt.show()
