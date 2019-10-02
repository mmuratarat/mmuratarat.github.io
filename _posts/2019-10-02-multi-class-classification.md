---
layout: post
title: "Multiclass Classification"
author: "MMA"
comments: true
---

Although many classification problems can be defined using two classes (they are inherently multi-class classifiers), some are defined with more than two classes which requires adaptations of machine learning algorithm.

Logistic Regression can be naturally extended to multi-class learning problems by replacing the sigmoid function with the softmax function. The KNN algorithm is also straightforward to extend to multiclass case. When we find $k$ closest examples using a distance metric such as Euclidean Distance, for the input $x$ and examine them, we return the class that we saw the most among the $k$ examples. Multi-class labeling is also trivial with Naive Bayes classifier. For each test example $i$,�, and each, class $c$, you want to find $\arg \max_{c} P(\textrm{class}_{c} | \textrm{data}_{i})$

SVM cannot be naturally extended to multi-class problems. Other algorithms can be implemented more efficiently in the binary case. What should you do if you have a multi-class problem but a binary classification learning algorithm?

One common strategy is called One-vs-All (usually referred to as One-versus-Rest or OVA classification). The idea is to transform a multi-class problem into C binary classification problem and build C different binary classifiers. Here, you pick one class and train a binary classifier with the samples of selected class on one side and other samples on the other side. Thus, you end up with C classifiers. While testing, you simply classify the sample as belonging to the class with maximum score among C classifier. For example, if we have three classes, $y \in \\{1, 2, 3\\}$, we create copies of the original dataset and modify them. In the first copy, we replace all labels not equal to 1 by 0. In the second copy, we replace all labels not equal to 2 by 0. In the third copy, we replace all labels not equal to 3 by 0. Now we have three binary classification problems where we have to learn to distinguish between labels 1 and 0; 2 and 0; and 3 and 0. 

Another strategy is One-vs-One (OVO, also known as All-versus-All or AVA). Here, you pick 2 classes at a time and train a binary classifier using samples from the selected two-classes only (other samples are ignored in this step). You repeat this for all the two-class combinations. So, you end up with $\frac{C (C-1)}{2}$ number of classifiers. At prediction time, a voting scheme is applied: all $C (C − 1) / 2$ classifiers are applied to an unseen sample and the class that got the highest number of "+1" predictions gets predicted by the combined classifier. All-versus-all tends to be superior to one-versus-all.

A problem with the previous schemes is that binary classifiers are sensitive to errors. If any classifier makes an error, it can affect the vote count.

In One-vs-One scheme, each individual learning problem only involves a small subset of data whereas with One-vs-All, the complete dataset is used `number of classes` times.

## OneVsRestClassifier of Sci-kit Learn

{% highlight python %} 
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.testing import assert_equal

iris = datasets.load_iris()
X, y = iris.data, iris.target
ovr = OneVsRestClassifier(LinearSVC(random_state=0, multi_class='ovr')).fit(X, y)
# For the answer of why OneVsRestClassifier wrapper and multi_class='ovr', look below!

# I used LinearSVC() instead of SVC(kernel='linear') for this particular problem.
# They are basically the same just with some implementation differences according to docs.
# https://stackoverflow.com/questions/45384185/what-is-the-difference-between-linearsvc-and-svckernel-linear

ovr.n_classes_
#number of classes
#3

ovr.estimators_
#Number of estimators must equal to num_classes

# [LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#            multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#            verbose=0),
#  LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#            multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#            verbose=0),
#  LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#            multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#            verbose=0)]

ovr.predict(X) 
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
#        2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

y
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
#        2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

ovr.score(X, y)
#Returns the mean accuracy on the given test data and labels.
#0.9666666666666667

#decision_function is only used with a SVM classifier reason being it gives out the distance of your data points from the hyperplane that separates the data

# How to interpret decision function?
# https://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict/20114601
ovr.decision_function(X)
#The desion function tells us on which side of the hyperplane generated by the classifier we are (and how far we are away from it). 
#Based on that information, the estimator then label the examples with the corresponding label.
{% endhighlight %}

`OneVsRestClassifier` is designed to model each class against all of the other classes independently, and create a classifier for each situation. The way I understand this process is that `OneVsRestClassifier` grabs a class, and creates a binary label for whether a point is or isn't that class. Then this labelling gets fed into whatever estimator you have chosen to use. I believe the confusion comes in in that SVC also allows you to make this same choice, but in effect with this implementation the choice will not matter because you will always only be feeding two classes into the SVC.

1. https://stackoverflow.com/a/43506826/1757224
2. https://stackoverflow.com/questions/39604468/what-is-the-difference-between-onevsrestclassifier-with-svc-and-svc-with-decisio?rq=1


#### REFERENCES
1. [https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html){:target="_blank"}
2. [https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html){:target="_blank"}
3. [https://scikit-learn.org/stable/modules/multiclass.html](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
4. [http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2015/lecture9.pdf](http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2015/lecture9.pdf){:target="_blank"}
5. [https://gemfury.com/stream/python:scikit-learn/-/content/tests/test_multiclass.py](https://gemfury.com/stream/python:scikit-learn/-/content/tests/test_multiclass.py){:target="_blank"}