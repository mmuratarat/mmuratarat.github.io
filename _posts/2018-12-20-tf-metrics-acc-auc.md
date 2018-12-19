---
layout: post
title:  "Tensorflow Metrics - Accuracy/AUC"
author: "MMA"
---

## tf.metrics
Tensorflow has many built-in evaluation-related metrics which can be seen [here](https://www.tensorflow.org/api_docs/python/tf/metrics){:target="_blank"}. However, sometimes, it can be hard to use them. In this post, I will briefly talk about accuracy and AUC measures. Calculation both metrics can be tricky and a bit counter-intuitive because both maintain a running measurements.

# tf.metrics.accuracy
`tf.metrics.accuracy` calculates how often predictions matches labels. `tf.metrics.accuracy` has many arguments and in the end returns two tensorflow operations: accuracy value and an update operation (whose purpose is to collect samples and build up your statistics). Two running variables are created and placed into the computational graph: `total` and `count` which are equivalent to number of correctly classified observations and number of observations, respectively. 

How to use `tf.metrics.accuracy` is pretty straightforward:

{% highlight python %}
labels = ...
predictions = ...

accuracy, update_op_acc = tf.metrics.accuracy(labels, predictions, name = 'accuracy')

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print("tf accuracy: {}".format(sess.run([accuracy, update_op_acc])))
{% endhighlight %}

However, one needs to be sure to initialize and/or reset the running variables correctly. `tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES` will print all the local variables. However, resetting local variables by running `sess.run(tf.local_variables_initializer())` can be a terrible idea because one might accidentally reset other local variables unintentionally. By being explicit about which variables to reset, we can avoid having troubles later with other local variables in our computational graph. Two running variables created for `tf.metrics.accuracy`  and can be called using `scope` argument of `tf.get.collection()`.

{% highlight python %}
running_vars_auc = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy')
{% endhighlight %}

`running_vars_auc` will contain:

{% highlight python %}
<tf.Variable 'accuracy/total:0' shape=() dtype=float32_ref>,
<tf.Variable 'accuracy/count:0' shape=() dtype=float32_ref>
{% endhighlight %}
Now, we need to specify an operation that will perform the initialization/resetting of those running variables.

{% highlight python %}
running_vars_auc_initializer = tf.variables_initializer(var_list=running_vars_auc)
{% endhighlight %}

Now, we need to this operation in out Tensorflow session in order to initialize/re-initalize/reset local variables of `tf.metrics.accuracy`, using:

{% highlight python %}
session.run(running_vars_auc_initializer)
{% endhighlight %}

# Full Example 
{% highlight python %}
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

labels = np.array([[1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0]], dtype=np.uint8)

predictions = np.array([[1,0,0,0],
                        [1,1,0,0],
                        [1,1,1,0],
                        [0,1,1,1]], dtype=np.uint8)

accuracy, update_op_acc = tf.metrics.accuracy(labels, predictions, name = 'accuracy')
print(accuracy)
#Tensor("accuracy/value:0", shape=(), dtype=float32)
print(update_op_acc)
#Tensor("accuracy/update_op:0", shape=(), dtype=float32)
tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
#[<tf.Variable 'accuracy/total:0' shape=() dtype=float32_ref>,
# <tf.Variable 'accuracy/count:0' shape=() dtype=float32_ref>]

running_vars_auc = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy')
running_vars_auc_initializer = tf.variables_initializer(var_list=running_vars_auc)


with tf.Session() as sess:
    sess.run(running_vars_auc_initializer)
    print("tf accuracy/update_op: {}".format(sess.run([accuracy, update_op_acc])))
    #tf accuracy/update_op: [0.0, 0.6875]
    print("tf accuracy: {}".format(sess.run(accuracy)))
    #tf accuracy: 0.6875
{% endhighlight %}

** NOTE: ** One must be careful how to reset variables when processing the data in batches. Arranging the operations while calculating overall accuracy and batch accuracy is different. One needs to reset the running variables to zero before calculating accuracy values of each new batch of data.

# tf.metrics.auc

Using `tf.metrics.auc` is completely similar. It computes the approximate AUC via a Riemann sum. `tf.metrics.auc` has many arguments and in the end returns two tensorflow operations: AUC value and an update operation. Four running variables are created and placed into the computational graph: `true_positives`, `true_negatives`, `false_positives` and `false_negatives`.

One must be aware that AUC result of `tf.metrics.auc` does not necessarily need to be matched with sklearn's because Tensorflow AUC is an approximate AUC via a Riemann sum. So, one might expect that the results will be a slightly different.

# Full Example
{% highlight python %}
# import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

print("sklearn auc: {}".format(roc_auc_score(y_true, y_scores)))
#sklearn auc: 0.75


auc, update_op_auc = tf.metrics.auc(y_true, y_scores, name = 'AUC')
print(auc)
#Tensor("auc/value:0", shape=(), dtype=float32)
print(update_op_auc)
#Tensor("auc/update_op:0", shape=(), dtype=float32)
print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
#[<tf.Variable 'AUC/true_positives:0' shape=(200,) dtype=float32_ref>, 
# <tf.Variable 'AUC/false_negatives:0' shape=(200,) dtype=float32_ref>, 
# <tf.Variable 'AUC/true_negatives:0' shape=(200,) dtype=float32_ref>, 
# <tf.Variable 'AUC/false_positives:0' shape=(200,) dtype=float32_ref>]


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print("tf auc/update_op: {}".format(sess.run([auc, update_op_auc])))
    #tf auc/update_op: [0.74999976, 0.74999976]
    print("tf auc: {}".format(sess.run(auc)))
    #tf auc: 0.7499997615814209
{% endhighlight %}

** NOTE ** Tensorflow's AUC metric supports only binary classification. Its first argument is `labels` which is Tensor whose shape matches predictions and will be cast to bool. Its second argument is is `predictions` which is a floating point Tensor of arbitrary shape and whose values are in the range `[0, 1]`.  That is, each element in `labels` states whether the class is "positive" or "negative" for a single observation. It is not a 1-hot encoded vector. Therefore, if you use softmax layer at the end of network, you can slice the predictions tensor to only consider the positive (or negative) class, which will represent the binary class:

{% highlight python %}
auc_value, auc_op = tf.metrics.auc(labels, predictions[:, 1])
{% endhighlight %}

## Links
1. [https://stackoverflow.com/a/46414395/1757224](https://stackoverflow.com/a/46414395/1757224){:target="_blank"}
2. [http://ronny.rest/blog/post_2017_09_11_tf_metrics/](http://ronny.rest/blog/post_2017_09_11_tf_metrics/){:target="_blank"}
3. [https://stackoverflow.com/a/50746989/1757224](https://stackoverflow.com/a/50746989/1757224){:target="_blank"}
