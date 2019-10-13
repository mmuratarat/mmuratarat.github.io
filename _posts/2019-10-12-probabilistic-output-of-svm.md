---
layout: post
title: "Can you interpret probabilistically the output of a Support Vector Machine?"
author: "MMA"
comments: true
---

SVMs don't output probabilities natively, but probability calibration methods can be used to convert the output to class probabilities. Various methods exist, including Platt scaling (particularly suitable for SVMs) and isotonic regression.

One standard way to obtain a "probability" out of an SVM is to use **Platt scaling**, which is available in many decent SVM implementations. In the binary case, the probabilities are calibrated using Platt scaling: logistic regression on the SVM's scores, fit by an additional cross-validation on the training data. 

Consider the problem of binary classification: for inputs $x$, we want to determine whether they belong to one of two classes, arbitrarily labeled $+1$ and $-1$. We assume that the classification problem will be solved by a real-valued function $f$, by predicting a class label $y = sign(f(x))$. For many problems, it is convenient to get a probability $P(y=1 \mid x)$, i.e. a classification that not only gives an answer, but also a degree of certainty about the answer. Some classification models do not provide such a probability, or give poor probability estimates.

Platt scaling is an algorithm to solve the aforementioned problem. It produces probability estimates

$$
P(y=1 \mid x)=\frac{1}{1+\exp(Af(x)+B)}
$$

i.e., a logistic transformation of the classifier scores $f(x)$, where $A$ and $B$ are two scalar parameters that are learned by the algorithm. 

You essentially create a new data set that has the same labels, but with one dimension (the output of the SVM). You then train on this new data set, and feed the output of the SVM as the input to this calibration method, which returns a probability. In Platt’s case, we are essentially just performing logistic regression on the output of the SVM with respect to the true class labels.

Note that predictions can now be made according to $y = 1$ iff $P(y=1 \mid x) > \frac{1}{2}$; if $B \neq 0$, the probability estimates contain a correction compared to the old decision function $y = sign(f(x))$. 

The parameters $A$ and $B$ are estimated using a maximum likelihood method that optimizes on the same training set as that for the original classifier $f$. To avoid overfitting to this set, a held-out calibration set or cross-validation can be used, but Platt additionally suggests transforming the labels $y$ to target probabilities:

* $t_{+} = \frac{N_{+} + 1}{N_{+} + 2}$ for positive samples ($y = 1$), and
* $t_{-} = \frac{1}{N_{-} + 2}$ for negative samples ($y = -1$).

Here, $N_{+}$ and $N_{-}$ are the number of positive and negative samples, respectively. This transformation follows by applying Bayes' rule to a model of out-of-sample data that has a uniform prior over the labels. The constants $1$ and $2$, on the numerator and denominator respectively, are derived from the application of Laplace Smoothing.

An alternative approach to probability calibration is to fit an isotonic regression model to an ill-calibrated probability model. The idea is to fit a piecewise-constant non-decreasing function instead of logistic regression. Piecewise-constant non-decreasing means stair-step shaped. This has been shown to work better than Platt scaling, in particular when enough training data is available.

a Relevance Vector Machine (RVM) can also be used instead of a plain SVM for probabilistic output. RVM is a machine learning technique that uses Bayesian inference to obtain parsimonious solutions for regression and probabilistic classification. The RVM has an identical functional form to the support vector machine, but provides probabilistic classification.

However, keep in mind that the probability estimates obtained with Platt scaling may be inconsistent with the prediction (look [here](https://scikit-learn.org/dev/modules/svm.html#scores-and-probabilities)).

Effectively, Platt scaling trains a probability model on top of the SVM's outputs under a cross-entropy loss function. To prevent this model from overfitting, It uses an internal five-fold cross validation. Needless to say, the cross-validation involved in Platt scaling is an expensive operation for large datasets. 

Additionally, the probability estimates may be inconsistent with the scores, in the sense that the "argmax" of the scores may not be the argmax of the probabilities. (E.g., in binary classification, a sample may be labeled by `predict` as belonging to a class that has probability $< \frac{1}{2}$ according to `predict_proba`.) 

For example, the $B$ parameter ("intercept" or "bias") can cause predictions based on probability estimates from this model to be inconsistent with the ones you get from the SVM decision function $f$. Suppose that $f(X) = 10$, then the prediction for $X$ is positive (because it has a positive sign); but if $B = - 9.9$ and $A = 1$, then $P(y \mid X) = 0.475$. 

Platt's method is also known to have theoretical issues. If confidence scores are required, but these do not have to be probabilities, then it is advisable to set `probability=False` and use `decision_function` instead of `predict_proba`.

Scikit-learn uses LibSVM internally, and this in turn uses Platt scaling, as detailed in [this note](https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf){:target="_blank"} by the LibSVM authors, to calibrate the SVM to produce probabilities in addition to class predictions.

### Additional notes

scikit-learn provides `CalibratedClassifierCV` which can be used to solve this problem: it allows to add probability output to LinearSVC or any other classifier which implements decision_function method:

`svm = LinearSVC()
 clf = CalibratedClassifierCV(svm) 
 clf.fit(X_train, y_train)
 y_proba = clf.predict_proba(X_test)`
 
[User guide](https://scikit-learn.org/stable/modules/calibration.html){:target="_blank"} has a nice section on that. By default `CalibratedClassifierCV` + `LinearSVC` will get you Platt scaling, but it also provides other options (isotonic regression method), and it is not limited to SVM classifiers.

