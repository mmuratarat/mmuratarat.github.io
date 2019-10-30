---
layout: post
title: "Softmax Regression"
author: "MMA"
comments: true
---

Softmax Regression (a.k.a. Multinomial Logistic, Maximum Entropy Classifier, or just Multi-class Logistic Regression) is a generalization of logistic regression that we can use for multi-class classification (under the assumption that the classes are mutually exclusive). In contrast, we use the (standard) Logistic Regression model in binary classification tasks.

The idea is quite simple: when given an instance $\mathbf{x}$, the Softmax Regression model first computes a score $s_{k}(\mathbf{x})$ for each class k, then estimates the probability of each class by applying the softmax function (also called the normalized exponential) to the scores. 

$$
s_{k}(\mathbf{x}) = \theta_{(k)} ^{T} \cdot \mathbf{x},\,\,\, k = 1,2,...,K
$$

Note that each class has its own dedicated parameter vector $\theta^{(k)}$. All these vectors are typically stored as rows in a parameter matrix.

Once you have computed the score of every class for instance $\mathbf{x}$, you can estimate the probability $\hat{p}_{k}$ that the instance belongs to class $k$ by running the scores through the softmax function: It computes the exponential of every score, then normalizes them (dividing by the sum of all exponentials).

$$
P(Y=k) = \dfrac{exp(s_{k}(\mathbf{x}))}{\sum_{j=1}^{K} exp(s_{j}(\mathbf{x}))},\,\,\, k = 1,2,...,K
$$

* K is the number of classes.
* $s(x)$ is a vector containing the scores of each class for the instance $\mathbf{x}$.
* Output of this function is the estimated probability that the instance $\mathbf{x}$ belongs to class $k$ given the scores of each class for that instance.

Just like the Logistic Regression classifier, the Softmax Regression classifier predicts the class with the highest estimated probability (which is simply the class with the highest score),

While training, we will be using [categorical cross entropy](https://mmuratarat.github.io/2018-12-21/cross-entropy#categorical-cross-entropy){:target="_blank"} and try to minimize it. Remember that when there are just two classes ($K = 2$), this cost function is equivalent to the Logistic Regression’s cost function which is binary cross entropy function, also called log-loss.

The Softmax Regression classifier predicts only one class at a time (i.e., multiclass, not multioutput) so it should be used only with mutually exclusive classes such as different types of plants. You cannot use it to recognize multiple people in a photograph.

{% highlight python %} 
# Load libraries
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')

# Train model
model = clf.fit(X_std, y)

#Intercept (a.k.a. bias) added to the decision function
#intercept_ : array, shape (n_classes,)
model.intercept_
#array([-0.20524097,  2.07483974, -1.86959878])

#Coefficient of the features in the decision function.
#coef_ : array, (n_classes, n_features)
model.coef_
# array([[-1.07406583,  1.16011506, -1.93069175, -1.81155649],
#        [ 0.58781008, -0.36184058, -0.36343111, -0.82626941],
#        [ 0.48625575, -0.79827448,  2.29412286,  2.6378259 ]])

#Logits (unscaled scores) for the first observation
np.multiply(X_std[0], model.coef_).sum(axis=1).T.reshape(1,3) + model.intercept_.T.reshape(1,3)
#array([[ 6.91487814,  2.75068455, -9.66556269]])
#It equals to the output of clf.decision_function(X_std)[0]

def softmax(w):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers

    Return
    ------
    a list of the same length as w of non-negative numbers
    """
    e = np.exp(np.array(w))
    softmax_result = e / np.sum(e)
    return softmax_result

softmax(np.multiply(X_std[0], model.coef_).sum(axis=1).T.reshape(1,3) + model.intercept_.T.reshape(1,3))
#array([[9.84695560e-01, 1.53043779e-02, 6.20166095e-08]])
#It is equal to the output of clf.predict_proba(X_std)[0]

# Create new observation
new_observation = [[.5, .5, .5, .5]]

# Predict class
model.predict(new_observation)
#array([1])

# View predicted probabilities
model.predict_proba(new_observation)
#array([[0.01982536, 0.74491994, 0.2352547 ]])
{% endhighlight %}

# REFERENCES
1. [https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html](https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html){:target="_blank"}
2. [https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html](https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html){:target="_blank"}
