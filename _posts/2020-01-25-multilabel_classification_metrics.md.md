---
layout: post
title: "Metrics for Multilabel Classification"
author: "MMA"
comments: true
---
Most of the supervised learning algorithms focus on either binary classification or multi-class classification. But sometimes, we will have dataset where we will have multi-labels for each observations. In this case, we would have different metrics to evaluate the algorithms, itself because multi-label prediction has an additional notion of being *partially correct*. 

Let's say that we have 4 observations and the actual and predicted values have been given as follows:

{% highlight python %} 
import numpy as np

y_true = np.array([[0,1,0],
                   [0,1,1],
                   [1,0,1],
                   [0,0,1]])

y_pred = np.array([[0,1,1],
                   [0,1,1],
                   [0,1,0],
                   [0,0,0]])
{% endhighlight %}

There are multiple example-based metrics to be used. We will look at couple of them below.

* **Exact Match Ratio**

  One trivial way around would just to ignore partially correct (consider them incorrect) and extend the *accuracy* used in single label case for multi-label prediction. 
  
  \begin{equation}
  \text{Exact Match Ratio, MR} = \frac{1}{n} \sum_{i=1}^{n} I(y_{i} = \hat{y_{i}})
  \end{equation}
  
  where $I$ is the indicator function. Clearly, a disadvantage of this measure is that it does not distinguish between complete incorrect and partially correct which might be considered harsh. 
  
  {% highlight python %} 
  MR = np.all(y_pred == y_true, axis=1).mean()
  #0.25
  {% endhighlight %}
  
* **0/1 Loss**

  This metric is basically known as $1 - \text{Exact Match Ratio}$, where we calculate proportions of instances whose actual value is not equal to predicted value.
  
  \begin{equation}
  \text{0/1 Loss} = \frac{1}{n} \sum_{i=1}^{n} I\left(y_{i} \neq \hat{y_{i}} \right)
  \end{equation}

  {% highlight python %} 
  01Loss = np.any(y_true != y_pred, axis=1).mean()
  #0.75
  {% endhighlight %}

* **Accuracy**

  Accuracy for each instance is defined as the proportion of the predicted correct labels to the total number (predicted and actual) of labels for that instance. Overall accuracy is the average across all instances. It is less ambiguously referred to as the *Hamming score*. 
  
  \begin{equation}
  \text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \frac{\lvert y_{i} \cap \hat{y_{i}}\rvert}{\lvert y_{i} \cup \hat{y_{i}}\rvert}
  \end{equation}
  
  {% highlight python %} 
  def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]
    
  Accuracy(y_true, y_pred)
  #0.375
  {% endhighlight %}
  
* **Hamming Loss**

  It reports how many times on average, the relevance of an example to a class label is incorrectly predicted. Therefore, hamming loss takes into account the prediction error (an incorrect label is predicted) and missing error (a relevant label not predicted), normalized over total number of classes and total number of examples.
  
  \begin{equation}
  \text{Hamming Loss} = \frac{1}{n L} \sum_{i=1}^{n}\sum_{j=1}^{L} I\left( y_{i}^{j} \neq \hat{y}_{i}^{j} \right)
  \end{equation}
  
  where $I$ is the indicator function. Ideally, we would expect the hamming loss to be 0, which would imply no error; practically the smaller the value of hamming loss, the better the performance of the learning algorithm. 
  
  {% highlight python %}
  def Hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])
    
  Hamming_Loss(y_true, y_pred)
  #0.4166666666666667
  {% endhighlight %}
  
* **Precision**

  It is the propotion of predicted correct labels to the total number of actual labels, averaged over all instances.
  
  \begin{equation}
  \text{Precision} = \frac{1}{n} \sum_{i=1}^{n} \frac{\lvert y_{i} \cap \hat{y_{i}}\rvert}{\lvert y_{i}\rvert}
  \end{equation}
  
  {% highlight python %}
  def Precision(y_true, y_pred):
      temp = 0
      for i in range(y_true.shape[0]):
          if sum(y_true[i]) == 0:
              continue
          temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
      return temp/ y_true.shape[0]
   #0.5
   {% endhighlight %}
  
* **Recall**

  It is the propotion of predicted correct labels to the total number of predicted labels, averaged over all instances.
  
  \begin{equation}
  \text{Recall} = \frac{1}{n} \sum_{i=1}^{n} \frac{\lvert y_{i} \cap \hat{y_{i}}\rvert}{\lvert \hat{y_{i}}\rvert}
  \end{equation}
  
  {% highlight python %}
  def Recall(y_true, y_pred):
      temp = 0
      for i in range(y_true.shape[0]):
          if sum(y_pred[i]) == 0:
              continue
          temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
      return temp/ y_true.shape[0]
   #0.375
   {% endhighlight %}

* **F1-Measure**

  Definition of *precision* and *recall* naturally leads to the following definition for F1-measure (harmonic mean of precision and recall):
  
  \begin{equation}
  F_{1} = \frac{1}{n} \sum_{i=1}^{n} \frac{2 \lvert y_{i} \cap \hat{y_{i}}\rvert}{\lvert y_{i}\rvert + \lvert \hat{y_{i}}\rvert}
  \end{equation}
  
  {% highlight python %}
  def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]
    
    print(F1Measure(y_true, y_pred))
    #0.41666666666666663
   {% endhighlight %}
  
One can also use Scikit Learn's functions to compute accuracy and Hamming loss:

{% highlight python %}
                   [0,0,0]])
import sklearn.metrics

print('Exact Match Ratio: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
#Exact Match Ratio: 0.25

print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 
#Hamming loss: 0.4166666666666667

#"samples" applies only to multilabel problems. It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes 
#for each sample in the evaluation data, and returning their (sample_weight-weighted) average.

print('Recall: {0}'.format(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'))) 
#Recall: 0.375

print('Precision: {0}'.format(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
#Precision: 0.5

print('F1 Measure: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'))) 
#F1 Measure: 0.41666666666666663
{% endhighlight %}

[Sorower (2010)](https://pdfs.semanticscholar.org/6b56/91db1e3a79af5e3c136d2dd322016a687a0b.pdf) gives a nice overview for other metrics to be used:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Multilabel_metrics.png?raw=true)
