# FOR ACTIVATION FUNCTIONS


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 
model="4H_relu_CONJ"
hardtanh= open("C:\Users\MAHE\Desktop\Programs\Python\\"+model+"_micro.csv")
np_hardtanh= np.loadtxt(fname = hardtanh, delimiter = ',',dtype='double')
np_range= np.arange(25,5025,step=25)
print '\nrange: ',np_range
plt.figure(figsize=(8, 6))
plt.plot(np_range,np_hardtanh,
        label='HardTanh',
        linewidth=2) 
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 2500.0])
plt.ylim([0.6, 0.75])
plt.xlabel('Iterations ')
plt.ylabel('AUC Score (Macro)')
plt.title('AUC Scores for different Activation Functions')
plt.legend(loc="lower right")
plt.show()
