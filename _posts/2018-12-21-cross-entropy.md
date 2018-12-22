---
layout: post
title: "Cross Entropy for Tensorflow"
author: "MMA"
---
Cross entropy can be used to define a loss function (cost function) in machine learning and optimization. It is defined on probability distributions, not single values. It works for classification because classifier output is (often) a probability distribution over class labels. 

For discrete distributions $p$ and $q$, it's given as follows:
 
$$H(p, q) = -\sum_y p(y) \log q(y)$$
 
When the cross entropy loss is used with 'hard' class labels, what this really amounts to is treating $p$ as the conditional empirical distribution over class labels. This is a distribution where the probability is 1 for the observed class label and 0 for all others. $q$ is the conditional distribution (probability of class label, given input) learned by the classifier. For a single observed data point with input $x_0$ and class $y_0$, we can see that the expression above reduces to the standard log loss (which is averaged over all data points):

$$-\sum_y I\{y = y_0\} \log q(y \mid x_0) = -\log q(y_0 \mid x_0)$$

Here, $I\\{\cdot\\}$ is the indicator function, which is $1$ when its argument is true or $0$ otherwise. The sum is taken over the set of possible class labels.

The concept of cross entropy applies equally well to continuous distributions. But, it can't be used for regression models that output a point estimate (e.g. the conditional mean because the regression function is the conditional mean of $Y$ over $X$, $E[Y|X]$.) but it can be used for models that output a full probability distribution. 

If you have a model that gives the full conditional distribution (probability of output given input), you can use cross entropy as a loss function.

For continuous distributions $p$ and $q$, the cross entropy is defined as:
$$H(p, q) = -\int_{Y} p(y) \log q(y) dy$$

Just considering a single observed input/output pair $(x, y)$, $p$ would be the empirical conditional distribution (a delta function over the observed output value), and $q$ would be the modeled conditional distribution (probability of output given input). In this case, the cross entropy reduces to $-\log q(y \mid x)$. Summing over data points, this is just the negative log likelihood!


# WHAT IS A COST (LOSS) FUNCTION?


In simple terms, predictive modeling can be described as the mathematical problem of approximating a mapping function ($f$) from input variables ($X$) to output variables ($y$):  $f: X \rightarrow y$. This is called the problem of function approximation.Stated in other words, the model learns how to take $X$ (i.e. features, or independent variable(s)) in order to predict $y$ (the target, the response or the dependent variable).  

If $y$ is discrete/categorical variable, then this is classification problem. If $y$ is real number/continuous, then this is a regression problem.

The goal is to approximate mapping function as accurately as possible, which consists of some parameters, given the time and resources available.

Once the model learns these parameters, they can be used to compute estimated values of $y$ given new values of $X$. In other words, you can use these learned parameters to predict values of $y$ when you don’t know what $y$ is, i.e., one has a predictive model!

In predictive modeling, cost functions are used to estimate how badly models are performing. Put it simply, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between $X$ and $y$. This is typically expressed as a difference or distance between the predicted value and the actual value. The cost function (you may also see this referred to as loss or error) can be estimated by iteratively running the model to compare estimated predictions against "ground truth", i.e., the known values of y.

The objective here, therefore, is to find parameters, weights/biases or a structure that minimises the cost function.

# WHAT IS A LOGIT?
A logit (also called a score) is a raw unscaled value associated with a class before computing the probability. In terms of a neural network architecture, this means that a logit is an output of a dense (fully-connected) layer. 

# BINARY CROSS-ENTROPY
Binary cross-entropy (a.k.a. log-loss/logistic loss) is a special case of categorical cross entropy. Withy binary cross entropy, you can classify only two classes, With categorical cross entropy, you are not limited to how many classes your model can classify.

Binary cross entropy formula is as follows:

$$ L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \left[y_{i} \log (p_i) + (1-y_{i}) \log (1- p_{i}) \right]$$

where $i$ indexes samples/observations. In the simplest case, each $y$ and $p$ is a number, corresponding to a probability of one class (we already have 2 classes. we need to choose one of them).

# CATEGORICAL CROSS-ENTROPY
Multi-class cross entropy formula is as follows:

$$ L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]$$

where $i$ indexes samples/observations and $j$ indexes classes. Here, $y_{ij}$ and $p_{ij}$ are expected to be probability distributions over $K$ classes. In a neural network, $y_{ij}$ is one-hot encoded labels and $p_{ij}$ is scaled (softmax) logits. 

When $K=2$, one will get binary cross entropy formula. 

# TENSORFLOW IMPLEMENTATIONS



# DIFFERENCE BETWEEN OBJECTIVE FUNCTION, COST FUNCTION AND LOSS FUNCTION
From Section 4.3 in "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaaron Courville:
![Placeholder image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/assets/DL_CE.jpeg "Image with caption")

In this book, at least, loss and cost are the same.

In Andrew NG's words:
<blockquote>Finally, the loss function was defined with respect to a single training example. It measures how well you're doing on a single training example. I'm now going to define something called the cost function, which measures how well you're doing an entire training set. So the cost function J which is applied to your parameters W and B is going to be the average with one of the m of the sum of the loss function applied to each of the training examples and turn.</blockquote>

The terms *cost function* and *loss function* are synonymous, some people also call it *error function*. 

However, there are also some different definitions out there. The loss function computes the error for a single training example, while the cost function will be average over all data points.

# HOW TO COMPUTE CROSS ENTROPY FOR BINARY CLASSIFICATION?
<script src="https://gist.github.com/mmuratarat/3db39c59e0436ec4768f27a3ad524808.js"></script>

# HOW TO COMPUTE CROSS ENTROPY FOR MULTICLASS CLASSIFICATION?
<script src="https://gist.github.com/mmuratarat/b7469a36d88fa88056b8511d8b1aac26.js"></script>

# DIFFERENCE BETWEEN tf.nn.softmax_cross_entropy_with_logits AND tf.nn.sparse_softmax_cross_entropy_with_logits

The function arguments for `tf.losses.softmax_cross_entropy` and `tf.losses.sparse_softmax_cross_entropy` are different, however, they produce the same result. 

The difference is simple:

* For `sparse_softmax_cross_entropy_with_logits`, labels must have the shape `[batch_size]` and the dtype is int32 or int64. Each label is an integer in range `[0, num_classes-1]`.
* For `softmax_cross_entropy_with_logits`, labels must have the shape `[batch_size, num_classes]` and dtype is float32 or float64.

Labels used in `softmax_cross_entropy_with_logits` are the one hot version of labels used in `sparse_softmax_cross_entropy_with_logits`.

**NOTE:** `tf.losses.softmax_cross_entropy` creates a cross-entropy loss using `tf.nn.softmax_cross_entropy_with_logits_v2`. Similarly, `tf.losses.sparse_softmax_cross_entropy` creates cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`. Convenience is that using `tf.nn.softmax_cross_entropy_with_logits_v2` or `tf.nn.sparse_softmax_cross_entropy_with_logits`, one can calculate individual entropy values and then using `tf.reduce_mean`, the average of the entire training set can be found.

<script src="https://gist.github.com/mmuratarat/f295d1017bcbb54c2f9ac5cd6d9f762d.js"></script>

# LINKS
1. [https://scikit-learn.org/stable/modules/multiclass.html](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
2. [https://stackoverflow.com/a/47034889/1757224](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
3. [https://stats.stackexchange.com/a/327396/16534](https://stats.stackexchange.com/a/327396/16534){:target="_blank"}
4. [https://stackoverflow.com/a/48317496/1757224](https://stackoverflow.com/a/48317496/1757224){:target="_blank"}
5. [https://scikit-learn.org/stable/modules/multiclass.html](https://scikit-learn.org/stable/modules/multiclass.html){:target="_blank"}
6. [https://chrisyeh96.github.io/2018/06/11/logistic-regression.html](https://chrisyeh96.github.io/2018/06/11/logistic-regression.html){:target="_blank"}
7. [https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi](https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi){:target="_blank"}
8. [https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow](https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow){:target="_blank"}
9. [https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro](https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro){:target="_blank"}
10. [https://stackoverflow.com/questions/49044398/is-there-any-difference-between-cross-entropy-loss-and-logistic-loss](https://stackoverflow.com/questions/49044398/is-there-any-difference-between-cross-entropy-loss-and-logistic-loss){:target="_blank"}
11. [https://stackoverflow.com/a/37317322/1757224](https://stackoverflow.com/a/37317322/1757224){:target="_blank"}
12. [https://datascience.stackexchange.com/a/9408/54046](https://datascience.stackexchange.com/a/9408/54046){:target="_blank"}
13. [https://stats.stackexchange.com/a/215484/16534](https://stats.stackexchange.com/a/215484/16534){:target="_blank"}
14. [https://stats.stackexchange.com/a/215495/16534](https://stats.stackexchange.com/a/215495/16534){:target="_blank"}