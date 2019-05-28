---
layout: post
title: "How to get total number of parameters in Tensorflow"
author: "MMA"
comments: true
---

This is a function which gives the total number of parameters in Tensorflow:

{% highlight python %}
#TOTAL NUMBER OF PARAMETERS
total_parameters = 0
for variable in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
# shape is an array of tf.Dimension
    shape = variable.get_shape()
    print("Size of the matrix: {}".format(shape))
    print("How many dimensions it has: {}".format(len(shape)))
    variable_parameters = 1
    for dim in shape:
        print("Dimension: {}".format(dim))
        variable_parameters *= dim.value
    print("Total number of elements in a matrix: {}".format(variable_parameters))
    print("---------------------------------------------")
    total_parameters += variable_parameters
print("Total number of parameters: {}". format(total_parameters))
{% endhighlight %} 