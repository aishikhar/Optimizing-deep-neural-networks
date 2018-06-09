import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
# false_positive_rate 
# true_positive_rate 
flables_4H = open("C:\Users\MAHE\Desktop\Programs\Python\selected\lables4H.csv")
fpredicted_4H= open("C:\Users\MAHE\Desktop\Programs\Python\selected\predicted4H.csv")
fpredicted_6H_adam = open("C:\Users\MAHE\Desktop\Programs\Python\selected\predicted6H_adam.csv")
flables_6H_adam= open("C:\Users\MAHE\Desktop\Programs\Python\selected\lables6H_adam.csv")
flables_6H_cgd= open("C:\Users\MAHE\Desktop\Programs\Python\selected\lables_dbn.csv")
fpredicted_6H_cgd= open("C:\Users\MAHE\Desktop\Programs\Python\selected\predicted_dbn.csv")
flables_6H_sgd= open("C:\Users\MAHE\Desktop\Programs\Python\selected\lables_autoencoder.csv")
fpredicted_6H_sgd= open("C:\Users\MAHE\Desktop\Programs\Python\selected\predicted_autoencoder.csv")

y_test_4H= np.loadtxt(fname = flables_4H, delimiter = ',',dtype='double')
y_score_4H= np.loadtxt(fname = fpredicted_4H, delimiter = ',',dtype='double')
y_test_6H_adam= np.loadtxt(fname = flables_6H_adam, delimiter = ',',dtype='double')
y_score_6H_adam= np.loadtxt(fname = fpredicted_6H_adam, delimiter = ',',dtype='double')
y_test_6H_sgd= np.loadtxt(fname = flables_6H_sgd, delimiter = ',',dtype='double')
y_score_6H_sgd= np.loadtxt(fname = fpredicted_6H_sgd, delimiter = ',',dtype='double')
y_test_6H_cgd= np.loadtxt(fname = flables_6H_cgd, delimiter = ',',dtype='double')
y_score_6H_cgd= np.loadtxt(fname = fpredicted_6H_cgd, delimiter = ',',dtype='double')
"""
n_values_6H_sgd = np.max(y_test_6H_sgd) + 1
y_test_6H_sgd=np.eye(n_values_6H_sgd)[y_test_6H_sgd]
n_values_6H_sgd=np.max(y_score_6H_sgd)+1
y_score_6H_sgd=np.eye(n_values_6H_sgd)[y_score_6H_sgd]
n_values_6H_cgd = np.max(y_test_6H_cgd) + 1
y_test_6H_cgd=np.eye(n_values_6H_cgd)[y_test_6H_cgd]
n_values_6H_cgd=np.max(y_score_6H_cgd)+1
y_score_6H_cgd=np.eye(n_values_6H_cgd)[y_score_6H_cgd]
n_values_6H_adam = np.max(y_test_6H_adam) + 1
y_test_6H_adam=np.eye(n_values_6H_adam)[y_test_6H_adam]
n_values_6H_adam=np.max(y_score_6H_adam)+1
y_score_6H_adam=np.eye(n_values_6H_adam)[y_score_6H_adam]
n_values_4H = np.max(y_test_4H) + 1
y_test_4H=np.eye(n_values_4H)[y_test_4H]
n_values_4H=np.max(y_score_4H)+1
y_score_4H=np.eye(n_values_4H)[y_score_4H]
"""

#print 'y_predicted (4H): ',y_score_4H
#print 'y_test: (4H): ',y_test_4H

#print 'y_predicted (6H_adam):  ',y_score_6H_adam
#print 'y_test: (6H_adam): ',y_test_6H_adam

print 'y_predicted (6H_sgd):  ',y_score_6H_sgd
#print 'y_test: (6H_sgd): ',y_test_6H_sgd

#print 'y_predicted (6H_adam):  ',y_score_6H_cgd
#print 'y_test: (6H_adam): ',y_test_6H_cgd

n_classes=2
fpr_4H = dict()
tpr_4H = dict()
roc_auc_4H = dict()

fpr_6H_adam = dict()
tpr_6H_adam = dict()
roc_auc_6H_adam = dict()

fpr_6H_sgd = dict()
tpr_6H_sgd = dict()
roc_auc_6H_sgd = dict()

fpr_6H_cgd = dict()
tpr_6H_cgd = dict()
roc_auc_6H_cgd = dict()


for i in range(n_classes):
    fpr_4H[i], tpr_4H[i], _ = roc_curve(y_test_4H[:, i], y_score_4H[:, i])
    roc_auc_4H[i] = auc(fpr_4H[i],tpr_4H[i])

    fpr_6H_adam[i], tpr_6H_adam[i], _ = roc_curve(y_test_6H_adam[:, i], y_score_6H_adam[:, i])
    roc_auc_6H_adam[i] = auc(fpr_6H_adam[i],tpr_6H_adam[i])

    fpr_6H_sgd[i], tpr_6H_sgd[i], _ = roc_curve(y_test_6H_sgd[:, i], y_score_6H_sgd[:, i])
    roc_auc_6H_sgd[i] = auc(fpr_6H_sgd[i],tpr_6H_sgd[i])

    fpr_6H_cgd[i], tpr_6H_cgd[i], _ = roc_curve(y_test_6H_cgd[:, i], y_score_6H_cgd[:, i])
    roc_auc_6H_cgd[i] = auc(fpr_6H_cgd[i],tpr_6H_cgd[i])
    
# Compute micro-average ROC curve and ROC area
fpr_4H["micro"], tpr_4H["micro"], _ = roc_curve(y_test_4H.ravel(), y_score_4H.ravel())
roc_auc_4H["micro"] = auc(fpr_4H["micro"], tpr_4H["micro"])  

fpr_6H_adam["micro"], tpr_6H_adam["micro"], _ = roc_curve(y_test_6H_adam.ravel(), y_score_6H_adam.ravel())
roc_auc_6H_adam["micro"] = auc(fpr_6H_adam["micro"], tpr_6H_adam["micro"])    

fpr_6H_cgd["micro"], tpr_6H_cgd["micro"], _ = roc_curve(y_test_6H_cgd.ravel(), y_score_6H_cgd.ravel())
roc_auc_6H_cgd["micro"] = auc(fpr_6H_cgd["micro"], tpr_6H_cgd["micro"])   

fpr_6H_sgd["micro"], tpr_6H_sgd["micro"], _ = roc_curve(y_test_6H_sgd.ravel(), y_score_6H_sgd.ravel())
roc_auc_6H_sgd["micro"] = auc(fpr_6H_sgd["micro"], tpr_6H_sgd["micro"])    
# First aggregate all false positive rates

"""all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
"""
# Plot all ROC curves

plt.figure(figsize=(10, 9))
"""plt.plot(fpr_4H["micro"], tpr_4H["micro"],
        label='4 Hidden Layers,CGD : AUC = {0:0.2f}'
            ''.format(roc_auc_4H["micro"]),
        linewidth=3)"""
plt.plot(fpr_6H_cgd["micro"], tpr_6H_cgd["micro"],
        label='Deep Belief Network: AUC = {0:0.3f}'
            ''.format(roc_auc_6H_cgd["micro"]),
        linewidth=3)
plt.plot(fpr_6H_sgd["micro"], tpr_6H_sgd["micro"],
        label='Stacked Autoencoder: AUC = 0.835'
            ''.format(roc_auc_6H_sgd["micro"]),
        linewidth=3)
plt.plot(fpr_6H_adam["micro"], tpr_6H_adam["micro"],
        label='6 Hidden Layers,SGD,Adam: AUC = 0.825'
            ''.format(roc_auc_6H_adam["micro"]),
        linewidth=3)

"""plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
        linewidth=2)"""
# plt.text(0.9,0.5, ('F1 Score: %.2f' % score).lstrip('0'),
#        size=15, horizontalalignment='right')

"""for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                ''.format(i, roc_auc[i]))
"""
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR (1- Specificity)')
plt.ylabel('TPR (Sensitivity)')
# plt.title('ROC Curve (Models on PCA)')
plt.legend(loc="lower right")
plt.show()