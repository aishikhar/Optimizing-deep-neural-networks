import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy import interp
import csv
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
# false_positive_rate
# true_positive_rate

model="apnea_all_features_4layers_relu"
k=25
roc_auc_macro=np.empty([1,200])
roc_auc_micro=np.empty([1,200])
for j in range(1,40):
    fpredicted = open("predicted"+model+str(k)+".csv")
    flables= open("lables"+model+".csv")
    k=k+25
    y_test= np.loadtxt(fname = flables, delimiter = ',',dtype='double')
    y_score= np.loadtxt(fname = fpredicted, delimiter = ',',dtype='double')
    
    print '\nIteration number: ',k
    y_predic=preprocessing.Binarizer(threshold=0.5).fit_transform(y_score)
    #print 'y_score\n',y_score,'\ny_predic\n',y_predic
    accuracy=metrics.accuracy_score(y_test,y_predic)
    precision=metrics.precision_score(y_test,y_predic,average='macro')
    recall=metrics.recall_score(y_test,y_predic,average='macro')
    f1=metrics.f1_score(y_test,y_predic,average='macro')
    n_classes=2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i],tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])     
    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Storing in array for later save
    roc_auc_micro[0][j]=roc_auc["micro"]
    roc_auc_macro[0][j]=roc_auc["macro"]
    # Plot all ROC curves
    print'\nMETRICS\n\nAUC :',roc_auc["micro"],'\nF1 Score: ',f1,'\nAccuracy: ',accuracy,'\nPrecision: ',precision,'\nRecall: ',recall
    """plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
            label='ROC AUC = {0:0.2f}'
                ''.format(roc_auc["micro"]),
            linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            linewidth=2)
    # plt.text(0.9,0.5, ('F1 Score: %.2f' % score).lstrip('0'),
    #        size=15, horizontalalignment='right')

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Deep MLP with Stochastic Gradient Updater')
    plt.legend(loc="lower right")
    plt.show()
#Save ROC CURVE RESULTS
#np.savetxt(model+"_macro.csv",roc_auc_macro, delimiter=",")
#np.savetxt(model+"_micro.csv",roc_auc_micro, delimiter=",")"""