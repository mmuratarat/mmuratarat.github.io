---
layout: post
title: "How to plot ROC curve and compute AUC by hand"
author: "MMA"
comments: true
---

Assume we have a probabilistic, binary classifier such as logistic regression.

Before presenting the ROC curve (= Receiver Operating Characteristic curve), the concept of confusion matrix must be understood. When we make a binary prediction, there can be 4 types of outcomes:

* We predict 0 while the true class is actually 0: this is called a True Negative, i.e. we correctly predict that the class is negative (0). For example, an antivirus did not detect a harmless file as a virus .
* We predict 0 while the true class is actually 1: this is called a False Negative, i.e. we incorrectly predict that the class is negative (0). For example, an antivirus failed to detect a virus.
* We predict 1 while the true class is actually 0: this is called a False Positive, i.e. we incorrectly predict that the class is positive (1). For example, an antivirus considered a harmless file to be a virus.
* We predict 1 while the true class is actually 1: this is called a True Positive, i.e. we correctly predict that the class is positive (1). For example, an antivirus rightfully detected a virus.

To get the confusion matrix, we go over all the predictions made by the model, and count how many times each of those 4 types of outcomes occur:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-01%20at%2010.48.20.png?raw=true)

Since to compare two different models it is often more convenient to have a single metric rather than several ones, we compute two metrics from the confusion matrix, which we will later combine into one:

* True positive rate (TPR), aka. sensitivity, hit rate, and recall, which is defined as $\frac{TP}{TP+FN}$. This metric corresponds to the proportion of positive data points that are correctly considered as positive, with respect to all positive data points. In other words, the higher TPR, the fewer positive data points we will miss.

* False positive rate (FPR), aka. fall-out, which is defined as $\frac{FP}{FP+TN} = 1 - \text{specificity}$. Intuitively this metric corresponds to the proportion of negative data points that are mistakenly considered as positive, with respect to all negative data points. In other words, the higher FPR, the more negative data points will be missclassified.

To combine the FPR and the TPR into one single metric, we first compute the two former metrics with many different threshold (for example 0.00;0.01,0.02,…,1.00) for the logistic regression, then plot them on a single graph, with the FPR values on the abscissa and the TPR values on the ordinate. The resulting curve is called ROC curve, and the metric we consider is the AUC of this curve, which we call AUROC. Threshold values from 0 to 1 are decided based on the number of samples in the dataset. 

The following figure shows the AUROC graphically:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/9NpXJ.png?raw=true)

AUC-ROC curve is basically the plot of sensitivity and 1 - specificity. It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity). It is a performance measurement (evaluation metric) for classification problems that consider all possible classification threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. The ROC curve is the probability that a classifier will be more confident that a randomly chosen positive example is actually positive than a randomly chosen negative example is positive.

In this figure, the blue area corresponds to the Area Under the curve of the Receiver Operating Characteristic (AUROC). The higher the area under the ROC curve, the better the classifier. The dashed line in the diagonal we present the ROC curve of a random predictor. It has an AUROC of 0.5. The random predictor is commonly used as a baseline to see whether the model is useful. A classifier with an AUC higher than 0.5 is better than a random classifier. If AUC is lower than 0.5, then something is wrong with your model. A perfect classifier would have an AUC of 1. Usually, if your model behaves well, you obtain a good classifier by selecting the value of threshold that gives TPR close to 1 while keeping FPR near 0. 

It is easy to see that if the threshold is zero, all our prediction will be positive, so both TPR and FPR will be 1. On the other hand, if the threshold is 1, then no positive prediction will be made, both TPR and FPR will be 0. 

For example, let's have a binary classification problem with 4 observations. We know true class and predicted probabilities obtained by the algorithm. All we need to do, based on different threshold values, is to compute True Positive Rate (TPR) and False Positive Rate (FPR) values for each of the thresholds and then plot TPR against FPR.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-01%20at%2011.09.09.png?raw=true)

You can obtain this table using the Pyhon code below:

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16,9)
%matplotlib inline

score = np.array([0.8, 0.6, 0.4, 0.2])
y = np.array([1,0,1,0])

# false positive rate
FPR = []
# true positive rate
TPR = []
# Iterate thresholds from 0.0, 0.01, ... 1.0
thresholds = np.arange(0.0, 1.01, 0.2)

# get number of positive and negative examples in the dataset
P = sum(y)
N = len(y) - P

# iterate through all thresholds and determine fraction of true positives
# and false positives found at this threshold
for thresh in thresholds:
    FP=0
    TP=0
    thresh = round(thresh,2) #Limiting floats to two decimal points, or threshold 0.6 will be 0.6000000000000001 which gives FP=0
    for i in range(len(score)):
        if (score[i] >= thresh):
            if y[i] == 1:
                TP = TP + 1
            if y[i] == 0:
                FP = FP + 1
    FPR.append(FP/N)
    TPR.append(TP/P)
{% endhighlight %}

When you obtain True Positive Rate and False Positive Rate for each of thresholds, all you need to is plot them!

{% highlight python %}
# This is the AUC
#you're integrating from right to left. This flips the sign of the result
auc = -1 * np.trapz(TPR, FPR)

plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve, AUC = %.2f'%auc)
plt.legend(loc="lower right")
plt.savefig('AUC_example.png')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/AUC_example.png?raw=true)