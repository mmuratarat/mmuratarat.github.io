---
layout: post
title: "Naive Bayes Classifier example by hand and how to do in Scikit-Learn"
author: "MMA"
comments: true
---

# Naive Bayes Classifier

A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.

$$
P(A \mid B) = \frac{P(A, B)}{P(B)} = \frac{P(B\mid A) \times P (A)}{P(B)}
$$

It is termed as 'Naive' because it assumes independence between every pair of feature in the data. That is presence of one particular feature does not affect the other. Bayes’ rule without the independence assumption is called a bayesian network. 


Let $(x_{1}, x_{2},..., x_{p})$ be a feature vector and $y$ be the class label corresponding to this feature vector. Applying Bayes' theorem,

$$
P(y \mid X) = \frac{P(X, y)}{P(X)} = \frac{P(X\mid y) \times P (y)}{P(X)}
$$

where $X$ is given as $X = (x_{1}, x_{2}, ..., x_{p})$. By substituting for X and expanding using the chain rule we get,

$$
P(y \mid x_{1}, x_{2},..., x_{p}) = \frac{P(x_{1}, x_{2},..., x_{p}, y)}{P(x_{1}, x_{2},..., x_{p})} = \frac{P(x_{1}, x_{2},..., x_{p}\mid y) \times P (y)}{P(x_{1}, x_{2},..., x_{p})}
$$


Since, $(x_{1}, x_{2},..., x_{p})$ are independent of each other,


$$
P(y \mid x_{1}, x_{2},..., x_{p}) = \frac{P (y) \times \prod_{i=1}^{p} P(x_{i} \mid y)}{\prod_{i=1}^{p} P(x_{i})}
$$

For all entries in the dataset, the denominator does not change, it remain static. Therefore, the denominator can be removed and a proportionality can be introduced.

$$
P(y \mid x_{1}, x_{2},..., x_{p}) \propto P (y) \times \prod_{i=1}^{p} P(x_{i} \mid y)
$$

In our case, the response variable ($y$) has only two outcomes, binary (e.g., yes or no / positive or negative). There could be cases where the classification could be multivariate. 

To complete the specification of our classifier, we adopt the MAP (Maximum A Posteriori) decision rule, which assigns the label to the class with the highest posterior.

$$
\hat{y} = \operatorname*{argmax}_{y} P (y) \times \prod_{i=1}^{p} P(x_{i} \mid y)
$$

We calculate probability for all 'k' classes using the above function and take one with the maximum value to classify a new point belongs to that class.

# Types of NB Classifier

1. **Multinomial Naive Bayes**: It is used for discrete counts. This is mostly used for document classification problem, i.e whether a document belongs to the category of sports, politics, technology etc. The features/predictors used by the classifier are the frequency of the words present in the document.

2. **Gaussian Naive Bayes**: It is used in classification and it assumes that the predictors/features take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution (follow a normal distribution). The parameters of the Gaussian are the mean and variance of the feature values. Since the way the values are present in the dataset changes, the formula for conditional probability changes to:

$$
P(x_{i} \mid y) = \frac{1}{\sqrt{2 \pi \sigma^{2}_{y}}} \exp \left(- \frac{\left(x_{i} - \mu_{y}\right)^{2}}{2\sigma^{2}_{y}} \right)
$$

If continuous features do not have normal distribution, we should use transformation or different methods to convert it in normal distribution.

Alternatively, a continuous feature could be discretized by binning its values, but doing so throws away information, and results could be sensitive to the binning scheme.

3. **Bernoulli Naive Bayes**: This is similar to the multinomial naive bayes but the predictors are boolean variables. The parameters that we use to predict the class variable take up only values yes or no, for example if a word occurs in the text or not.

# Advantages

* It is really easy to implement and often works well. Training is quick, and consists of computing the priors and the likelihoods. Prediction on a new data point is also quick. First calculate the posterior for each class. Then apply the MAP decision rule: the label is the class with the maximum posterior.

* CPU usage is modest: there are no gradients or iterative parameter updates to compute, since prediction and training employ only analytic formulae.

* The memory requirement is very low because these operations do not require the whole data set to be held in RAM at once.

* It is often a good first thing to try. For problems with a small amount of training data, it can achieve better results than other classifiers, thanks to Occam’s Razor because it has a low propensity to overfit.

* Easily handles missing feature values — by re-training and predicting without that feature! (See [this quora answer](https://www.quora.com/What-are-the-advantages-of-using-a-naive-Bayes-for-classification/answer/Muhammad-Zaheer-6?ch=10&share=fbcb0773&srid=CM1bE){:target="_blank"})

# Disadvantages

* Ensembling, boosting, bagging will not work here since the purpose of these methods is to reduce variance. Naive Bayes has no variance to minimize.

* It is important to note that categorical variables need to be factors with the same levels in both training and new data (testing dataset). This can be problematic in a predictive context if the new data obtained in the future does not contain all the levels, meaning it will be unable to make a prediction. This is often known as "Zero Frequency". To solve this, we can use a smoothing technique. One of the simplest smoothing techniques is called Laplace smoothing. 

* Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.

* It cannot incorporate feature interactions.

* Performance is sensitive to skewed data — that is, when the training data is not representative of the class distributions in the overall population. In this case, the prior estimates will be incorrect.

# An Example by hand

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/NB_example.png?raw=true)

Let's find the class of this test record:

$$
X=(Refund = No, Married, Income = 120K
$$

* For income variable when Evade = No:
   * Sample mean: 110
   * Sample Variance = 2975
* For income variable when Evade = Yes:
   * Sample mean: 90
   * Sample Variance = 25
   
Therefore,

$$
P(Income = 120 \mid Evade = No) =  \frac{1}{\sqrt{2 \times \pi \times 2975}} \exp \left(- \frac{\left(120 - 110\right)^{2}}{2 \times 2975} \right) = 0.007192295359419549
$$

$$
P(Income = 120 \mid Evade = Yes) =  \frac{1}{\sqrt{2 \times \pi \times 25}} \exp \left(- \frac{\left(120 - 90\right)^{2}}{2 \times 25} \right) = 1.2151765699646572e-09
$$

Let's compute the posterior probabilities.

$$
\begin{split}
P(X \mid Evade = No) &= P(Refund = No \mid Evade = No) \times P(Married \mid Evade = No) \times P(Income = 120K \mid Evade = No) \\
&= \frac{4}{7} \times \frac{4}{7} \times 0.007192295359419549 \\
&= 0.0023485046071574033
\end{split}
$$

$$
\begin{split}
P(X \mid Evade = Yes) &= P(Refund = No \mid Evade = Yes) \times P(Married \mid Evade = Yes) \times P(Income = 120K \mid Evade = Yes) \\
&= \frac{3}{3} \times 0 \times 1.2151765699646572e-09 \\
&= 0
\end{split}
$$

Since $P(X \mid Evade = No) \times P(Evade = No) > P(X \mid Evade = Yes) \times P(Evade = Yes)$, therefore, $P(No \mid X) > P(Yes \mid X)$, the class of this instance is then **No**.

# DATA: Iris Flower Dataset

{% highlight python %}
# loading libraries
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X = data['data']
y = data['target']
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
{% endhighlight %}

# Gaussian Naive Bayes Classifier in Sci-kit Learn

{% highlight python %}
# Fitting Naive Bayes Classification to the Training set with linear kernel
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

GNBclassifier = GaussianNB()
GNBclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_GNB = GNBclassifier.predict(X_test)

# evaluate accuracy
print('\nThe accuracy of Gaussian Naive Bayes Classifier is {}%'.format(accuracy_score(y_test, y_pred_GNB)*100))
#The accuracy of Gaussian Naive Bayes Classifier is 96.0%
{% endhighlight %}

# REFERENCES
1. [https://medium.com/@akshayc123/naive-bayes-classifier-nb-7429a1bdb2c0](https://medium.com/@akshayc123/naive-bayes-classifier-nb-7429a1bdb2c0){:target="_blank"}
2. [http://www.inf.u-szeged.hu/~ormandi/ai2/06-naiveBayes-example.pdf](http://www.inf.u-szeged.hu/~ormandi/ai2/06-naiveBayes-example.pdf){:target="_blank"}
3. [https://towardsdatascience.com/the-naive-bayes-classifier-e92ea9f47523](https://towardsdatascience.com/the-naive-bayes-classifier-e92ea9f47523){:target="_blank"}
