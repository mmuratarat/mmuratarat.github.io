---
title: "Implementing Softmax Function in Python"
author: "MMA"
layout: post
comments: true
---
The softmax function is used in various multiclass classification methods. It takes an un-normalized vector, and normalizes it into a probability distribution. It is often used in neural networks, to map the non-normalized output to a probability distribution over predicted output classes. It is a function which gets applied to a vector in $x \in R^{k}$ and returns a vector in $[0, 1]^{K}$ with the property that the sum of all elements is 1:

$$ \varphi(\mathbb{x})_j = \frac{e^{x_j}}{\sum_{k=1}^K e^{x_k}} \;\;\;\text{ for } j=1, \dots, K $$

# Python implementation

{% highlight python %}
import numpy as np

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

softmax([0.1, 0.2])
#array([0.47502081, 0.52497919])

softmax([-0.1, 0.2])
#array([0.42555748, 0.57444252])

softmax([0.9, -10])
#array([9.99981542e-01, 1.84578933e-05])

softmax([0, 10])
#array([4.53978687e-05, 9.99954602e-01])
{% endhighlight %}