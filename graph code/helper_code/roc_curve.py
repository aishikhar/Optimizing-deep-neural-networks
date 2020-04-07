import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score,f1_score,accuracy_score
from scipy import interp
# false_positive_rate 
# true_positive_rate 
fpredicted_svm = open("C:\Users\MAHE\Desktop\Programs\Python\predicted_smoteSVM.csv")
flables_svm= open("C:\Users\MAHE\Desktop\Programs\Python\labels_smoteSVM.csv")
fpredicted_rf = open("C:\Users\MAHE\Desktop\Programs\Python\predicted_smoteRandomForest.csv")
flables_rf= open("C:\Users\MAHE\Desktop\Programs\Python\labels_smoteRandomForest.csv")

y_test_svm= np.loadtxt(fname = flables_svm, delimiter = ',',dtype='double').astype(int)
y_score_svm= np.loadtxt(fname = fpredicted_svm, delimiter = ',',dtype='double').astype(int)
n_values_svm = np.max(y_test_svm) + 1
y_test_svm=np.eye(n_values_svm)[y_test_svm]
n_values_svm=np.max(y_score_svm)+1
y_score_svm=np.eye(n_values_svm)[y_score_svm]

y_test_rf= np.loadtxt(fname = flables_rf, delimiter = ',',dtype='double').astype(int)
y_score_rf= np.loadtxt(fname = fpredicted_rf, delimiter = ',',dtype='double').astype(int)
n_values_rf = np.max(y_test_rf) + 1
y_test_rf=np.eye(n_values_rf)[y_test_rf]
n_values_rf=np.max(y_score_rf)+1
y_score_rf=np.eye(n_values_rf)[y_score_rf]
print 'y_predicted (Random Forest): ',y_score_rf
print 'y_test: (Random Forest): ',y_test_rf

print 'y_predicted (SVM):  ',y_score_svm
print 'y_test: (SVM): ',y_test_svm


n_classes=2
fpr_svm = dict()
tpr_svm = dict()
roc_auc_svm = dict()

fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()


for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_rf[:, i], y_score_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i],tpr_rf[i])
    fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test_svm[:, i], y_score_svm[:, i])
    roc_auc_svm[i] = auc(fpr_svm[i],tpr_svm[i])
    
# Compute micro-average ROC curve and ROC area
fpr_rf["micro"], tpr_rf["micro"], _ = roc_curve(y_test_rf.ravel(), y_score_rf.ravel())
roc_auc_rf["micro"] = auc(fpr_rf["micro"], tpr_rf["micro"])    

fpr_svm["micro"], tpr_svm["micro"], _ = roc_curve(y_test_svm.ravel(), y_score_svm.ravel())
roc_auc_svm["micro"] = auc(fpr_svm["micro"], tpr_svm["micro"])    
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
plt.plot(fpr_rf["micro"], tpr_rf["micro"],
        label='Random Forest: AUC = {0:0.2f}'
            ''.format(roc_auc_rf["micro"]),
        linewidth=3)
plt.plot(fpr_svm["micro"], tpr_svm["micro"],
        label='Support Vector Machine: AUC = {0:0.2f}'
            ''.format(roc_auc_svm["micro"]),
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