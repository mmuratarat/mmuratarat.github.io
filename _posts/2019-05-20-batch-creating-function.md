---
layout: post
title: "Function to create batches of data"
author: "MMA"
comments: true
---

This is a basic function to create batches from a data set.

{% highlight python %}import math
import numpy as np

def miniBatch(x, y, batchSize):
    numObs  = x.shape[0]
    batches = [] 
    batchNum = math.floor(numObs / batchSize)
    
    if numObs % batchSize == 0:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
    else:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
        xBatch = x[batchNum * batchSize:, :]
        yBatch = y[batchNum * batchSize:, :]
        batches.append((xBatch, yBatch))
    return batches
{% endhighlight %} 

Here $x$ and $y$ should be numpy array. Let's create an example dataset and see how the function works:

{% highlight python %}from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

# Define dummy training data, N examples
N = 1000
# Create a simulated feature matrix and output vector with 100 samples,
X_train, output = make_classification(n_samples = N,
                                       # ten features
                                       n_features = n_features,
                                       # three output classes
                                       n_classes = n_classes,
                                       n_informative=4,
                                       n_redundant=2,
                                       # with 20% of observations in the first class, 30% in the second class,
                                       # and 50% in the third class [.2, .3, .8]. ('None' makes balanced classes)
                                       weights = None)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = output.reshape(len(output), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)

X_train.shape
#(1000, 32)
y_train.shape
#(1000, 8)

X_train
# array([[ 0.45396437,  1.51656331,  1.3629891 , ...,  0.14717609,
#         -0.15119727,  1.01995008],
#        [-0.81950039,  0.26249003, -0.51083559, ...,  0.36629063,
#          1.13654148, -0.19591568],
#        [ 1.31671553, -1.30269225, -0.56231267, ...,  0.40714641,
#         -1.05379444,  2.03267767],
#        ...,
#        [-0.31350026,  0.65172311, -1.21891585, ...,  0.64381309,
#          0.18544175,  1.74250682],
#        [-0.84452574,  0.12441828, -1.73681735, ...,  0.50854846,
#         -0.04233258,  0.51991385],
#        [-0.33021028, -0.32264822, -0.27926319, ..., -1.01789405,
#         -0.55645379,  2.15772776]])

y_train
# array([[0., 0., 0., ..., 0., 0., 0.],
#        [0., 1., 0., ..., 0., 0., 0.],
#        [0., 0., 1., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 1., 0., ..., 0., 0., 0.],
#        [0., 1., 0., ..., 0., 0., 0.]])
{% endhighlight %}

Here, if number of observations in the dataset is not divisible by batch size, the last batch should not be dropped in the case it has fewer than batch size elements.

{% highlight python %}miniBatches = miniBatch(X_train, y_train, batchSize=64)
total_batch = len(miniBatches)
# Loop over all batches
for batch in miniBatches:
    xBatch = batch[0]
    yBatch = batch[1]
    print(xBatch.shape)
    
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (64, 32)
# (40, 32)
{% endhighlight %}
