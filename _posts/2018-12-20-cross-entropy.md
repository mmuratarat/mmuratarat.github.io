---
title: "Cross Entropy for Tensorflow"
output: html_document
---

# WHAT IS A LOGIT?
A logit (also called a score) is a raw unscaled value associated with a class before computing the probability. In terms of a neural network architecture, this means that a logit is an output of a dense (fully-connected) layer. 

#BINARY CROSS-ENTROPY

$ L(/theta) = - \frac{1}{n} \sum_{i=1}^n}$

# CATEGORICAL CROSS-ENTROPY

Binary cross-entropy (a.k.a. log-loss/logistic loss) is a special case of categorical cross entropy. Withy binary cross entropy, you can classify only two classes, With categorical cross entropy, you are not limited to how many classes your model can classify.

#TENSORFLOW IMPLEMENTATIONS



# DIFFERENCE BETWEEN OBJECTIVE FUNCTION, COST FUNCTION AND LOSS FUNCTION
From Section 4.3 in "Deep Learning" by ian goodfellow, Yoshua Bengio, Aaaron Courville:
![Placeholder image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/assets/DL_CE.jpeg "Image with caption")

In this book, at least, loss and cost are the same.

In Andrew NG's words:
<blockquote>Finally, the loss function was defined with respect to a single training example. It measures how well you're doing on a single training example. I'm now going to define something called the cost function, which measures how well you're doing an entire training set. So the cost function J which is applied to your parameters W and B is going to be the average with one of the m of the sum of the loss function applied to each of the training examples and turn.</blockquote>

The terms cost and loss functions are synonymous some people also call it error function. 

However, there are also some different definitions. The loss function computes the error for a single training example, while the cost function is the average of the loss functions of the entire training set.

# HOW TO COMPUTE CROSS ENTROPU FOR BINARY CLASSIFICATION?
<script src="https://gist.github.com/mmuratarat/3db39c59e0436ec4768f27a3ad524808.js"></script>

# HOW TO COMPUTE CROSS ENTROPU FOR MULTICLASS CLASSIFICATION?
<script src="https://gist.github.com/mmuratarat/b7469a36d88fa88056b8511d8b1aac26.js"></script>

# DIFFERENCE BETWEEN tf.nn.softmax_cross_entropy_with_logits AND tf.nn.sparse_softmax_cross_entropy_with_logits

<script src="https://gist.github.com/mmuratarat/f295d1017bcbb54c2f9ac5cd6d9f762d.js"></script>

# LINKS
https://scikit-learn.org/stable/modules/multiclass.html
https://stackoverflow.com/a/47034889/1757224
https://stats.stackexchange.com/a/327396/16534
https://stackoverflow.com/a/48317496/1757224
https://scikit-learn.org/stable/modules/multiclass.html
https://chrisyeh96.github.io/2018/06/11/logistic-regression.html
https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi
https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow
https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
https://stackoverflow.com/questions/49044398/is-there-any-difference-between-cross-entropy-loss-and-logistic-loss