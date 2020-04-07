# FOR VARYING LAYERS IN MLP


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
import csv
# false_positive_rate 
# true_positive_rate 
model1="micro_relu_2layers"
model2="micro_relu_3layers"
model3="micro_relu_4layers"
model4="micro_relu_5layers"
model5="micro_relu_6layers"
model6="macro_relu_7layers"
model7="7H_RELU"
model8="8H_RELU"
layer1= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model1+".csv")
layer2= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model2+".csv")
layer3= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model3+".csv")
layer4= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model4+".csv")
layer5= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model5+".csv")
layer6= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model6+".csv")
layer7= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model7+"_macro.csv")
layer8= open("C:\Users\MAHE\Desktop\Programs\Python\old\\"+model8+"_macro.csv")

np_layer1= np.loadtxt(fname = layer1, delimiter = ',',dtype='double')
np_layer2= np.loadtxt(fname = layer2, delimiter = ',',dtype='double')
np_layer3= np.loadtxt(fname = layer3, delimiter = ',',dtype='double')
np_layer4= np.loadtxt(fname = layer4, delimiter = ',',dtype='double')
np_layer5= np.loadtxt(fname = layer5, delimiter = ',',dtype='double')
np_layer6= np.loadtxt(fname = layer6, delimiter = ',',dtype='double')
np_layer7= np.loadtxt(fname = layer7, delimiter = ',',dtype='double')
np_layer8= np.loadtxt(fname = layer8, delimiter = ',',dtype='double')

np_range= np.arange(25,5025,step=25)
plt.figure(figsize=(8, 6))
"""plt.plot(np_range,np_layer1,
        label='MLP 1 Hidden Layer',
        linewidth=2)
plt.plot(np_range,np_layer2,
        label='MLP 2 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer3,
        label='MLP 3 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer4,
        label='MLP 4 Hidden Layers',
        linewidth=2)
plt.plot(np_range,np_layer5,
        label='MLP 5 Hidden Layers',
        linewidth=2)"""
plt.plot(np_range,np_layer6,
        label='MLP 6 Hidden Layers',
        linewidth=2, color='#e5e500')
plt.plot(np_range,np_layer7,
        label='MLP 7 Hidden Layers',
        linewidth=2, color='black') 
plt.plot(np_range,np_layer8,
        label='MLP 8 Hidden Layers',
        linewidth=2, color= 'green')
# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 4000.0])
plt.ylim([0.4, 0.65])
plt.xlabel('Iterations')
plt.ylabel('AUC Scores')
plt.title('Varying Layers in Multi-Layer Perceptron Network')
plt.legend(loc="lower right")
plt.show()
