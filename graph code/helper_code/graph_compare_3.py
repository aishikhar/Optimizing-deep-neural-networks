# FOR OPTIMIZATION ALGORITHMS

# For AUC SCORE COMPARISON
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 

"""
model1="4H_relu_LINEAR"
model2="4H_relu_STOCH"
model3="4H_relu_CONJ"
lgd= open("C:\Users\MAHE\Desktop\Programs\Python\\"+model1+"_micro.csv")
sgd= open("C:\Users\MAHE\Desktop\Programs\Python\\"+model2+"_micro.csv")
cgd= open("C:\Users\MAHE\Desktop\Programs\Python\\"+model3+"_micro.csv")

np_lgd= np.loadtxt(fname = lgd, delimiter = ',',dtype='double')
np_sgd= np.loadtxt(fname = sgd, delimiter = ',',dtype='double')
np_cgd= np.loadtxt(fname = cgd, delimiter = ',',dtype='double')
np_range= np.arange(25,5025,step=25)

plt.figure(figsize=(8, 6))
plt.plot(np_range,np_sgd,
        label='Stochastic Gradient Descent',
        linewidth=2)
plt.plot(np_range,np_cgd,
        label='Conjugate Gradient Descent',
        linewidth=2)
plt.plot(np_range,np_lgd,
        label='Line Gradient Descent',
        linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1600.0])
plt.ylim([0.65, 0.70])
plt.xlabel('Epochs')
plt.ylabel('AUC Scores')
plt.title('Varying Optimization Algorithms')
plt.legend(loc="lower right")
plt.show()
"""

# For Training Error Rate Comparison

model1="4H_relu_LINEAR"
model2="4H_relu_STOCH"
model3="4H_relu_CONJ"
lgd= open("D:\documentation\\"+model1+"\\"+model1+"_scores.csv")
sgd= open("D:\documentation\\"+model2+"\\"+model2+"_scores.csv")
cgd= open("D:\documentation\\"+model3+"\\"+model3+"_scores.csv")

np_lgd= np.loadtxt(fname = lgd, delimiter = ',',dtype='double')
np_sgd= np.loadtxt(fname = sgd, delimiter = ',',dtype='double')
np_cgd= np.loadtxt(fname = cgd, delimiter = ',',dtype='double')
np_range= np.arange(0,5001,step=1)

plt.figure(figsize=(8, 6))
plt.plot(np_range,np_lgd,
        label='Line Gradient Descent',
        linewidth=2)
plt.plot(np_range,np_sgd,
        label='Stochastic Gradient Descent',
        linewidth=2)
plt.plot(np_range,np_cgd,
        label='Conjugate Gradient Descent',
        linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 150.0])
plt.ylim([0, 0.70])
plt.xlabel('Epochs')
plt.ylabel('Negative Log Likelihood Error')
plt.title('Varying Optimization Algorithms')
plt.legend(loc="upper right")
plt.show()