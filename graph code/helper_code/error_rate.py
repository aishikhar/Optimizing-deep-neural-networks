# FOR VARYING UPDATERS


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 

layer2= open("C:\Users\MAHE\Desktop\Programs\Python\old\scoresapnea_all_features_2layers_relu.csv")

layer3= open("C:\Users\MAHE\Desktop\Programs\Python\old\scoresapnea_all_features_3layers_relu.csv")

layer4= open("C:\Users\MAHE\Desktop\Programs\Python\old\scoresapnea_all_features_4layers_relu.csv")

layer5= open("C:\Users\MAHE\Desktop\Programs\Python\old\scoresapnea_all_features_5layers_relu.csv")

layer6= open("C:\Users\MAHE\Desktop\Programs\Python\old\scoresapnea_all_features_6layers_relu.csv")

layer7= open("C:\Users\MAHE\Desktop\Programs\Python\old\scoresapnea_all_features_7layers_relu.csv")

layer8= open("C:\Users\MAHE\Desktop\Programs\Python\old\\7H_RELU_scores.csv")
layer9= open("C:\Users\MAHE\Desktop\Programs\Python\old\\8H_RELU_scores.csv")


np_layer2= np.loadtxt(fname = layer2, delimiter = ',',dtype='double')
np_layer3= np.loadtxt(fname = layer3, delimiter = ',',dtype='double')
np_layer4= np.loadtxt(fname = layer4, delimiter = ',',dtype='double')
np_layer5= np.loadtxt(fname = layer5, delimiter = ',',dtype='double')
np_layer6= np.loadtxt(fname = layer6, delimiter = ',',dtype='double')
np_layer7= np.loadtxt(fname = layer7, delimiter = ',',dtype='double')
np_layer8= np.loadtxt(fname = layer8, delimiter = ',',dtype='double')
np_layer9= np.loadtxt(fname = layer9, delimiter = ',',dtype='double')


np_range= np.arange(0,np_layer2.shape[0],step=1)
plt.figure(figsize=(8, 6))
plt.plot(np_range,np_layer2,
        label='MLP 1 Hidden Layer',
        linewidth=2)
plt.plot(np_range,np_layer3,
        label='MLP 2 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer4,
        label='MLP 3 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer5,
        label='MLP 4 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer6,
        label='MLP 5 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer7,
        label='MLP 6 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer8,
        label='MLP 7 Hidden Layers',
        linewidth=2)
"""plt.plot(np_range,np_layer8,
        label='MLP 8 Hidden Layers',
        linewidth=2)"""

# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 4000.0])
plt.ylim([0.0, 0.05])
plt.xlabel('Iterations')
plt.ylabel('Negative Log-Likelihood Error')
plt.title('Varying Learning Rate Updater Algorithms')
plt.legend(loc="upper right")
plt.show()
