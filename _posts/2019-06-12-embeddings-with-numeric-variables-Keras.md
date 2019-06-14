---
layout: post
title: "How to use embedding layer and other feature columns together in a network using Keras?"
author: "MMA"
comments: true
---

# Why should you use an embedding layer? 

One-Hot encoding is a commonly used method for converting a categorical input variable into continuous variable. For every level present, one new variable will be created. Presence of a level is represent by 1 and absence is represented by 0. However, one-hot encoded vectors are high-dimensional and sparse. One-hot encoding of high cardinality features often results in an unrealistic amount of computational resource requirement. It also treats different values of categorical variables completely independent of each other and often ignores the informative relations between them.

For an example, considering days of the week in a dataset, this is how we create dummies for that particular column:
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/dummying.png?raw=true)

The vectors of each embedding, though, are learned while training the neural network. The embedding reduces the memory usage and speeds up the training comparing with one-hot encoding. More importantly though, this approach allows for relationships between categories to be captured. Perhaps Saturday and Sunday have similar behavior, and maybe Friday behaves like an average of a weekend and a weekday.
 
For instance, a 4-dimensional version of an embedding for some days of the week could look like:

{% highlight markdown %}
| Days    	| Embeddings       	|
|---------	|------------------	|
| Sunday  	| [.8, .2, .1, .1] 	|
| Monday  	| [.1, .2, .9, .9] 	|
| Tuesday 	| [.2, .1, .9, .8] 	|
{% endhighlight %}

Here, Monday and Tuesday are fairly similar, yet they are both quite different from Sunday. Again, this is a toy example.

The advantage of doing this compared to the traditional approach of creating dummy variables (i.e. doing one hot encodings), is that each day can be represented by four numbers instead of one, hence we gain higher dimensionality and much richer relationships. 

Another advantage of embeddings is that the learned embeddings can be visualized to show which categories are similar to each other. The most popular method for this is t-SNE, which is a technique for dimensionality reduction that works particularly well for visualizing data sets with high-dimensionality.
 
# Embedding Dimensionality
The embedding-size defines the dimensionality in which we map the categorical variables. Jeremy Howard provides a general rule of thumb about the number of embedding dimensions: embedding size = min(50, number of categories/2). This [Google Blog](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html){:target="_blank"} also tells that a good rule of thumb is 4th root of the number of categories. Therefore, So it’s kind of experimental. However, literature shows that embedding dimensions of size 50 produces the most accurate results.

# How to use embedding layer with numeric variables?
Using embeddings with numeric variables is pretty straightforward. In order to combine the categorical data with numerical data, the model should use multiple inputs using [Keras functional API](https://keras.io/getting-started/functional-api-guide/){:target="_blank"}. One for each categorical variable and one for the numerical inputs. For the other non-categorical data columns, we simply send them to the model like we would do for any regular network. So once we have the individual models merged into a full model, we can add layers on top of it network and train it.

{% highlight markdown %}
   multi-hot-encode-input  num_data_input 
            |                   |
            |                   |
            |                   |
        embedding_layer         |
            |                   |
            |                   | 
             \                 /        
               \              / 
              dense_hidden_layer
                     | 
                     | 
                  output_layer 
{% endhighlight %}

{% highlight python %}
import tensorflow as tf
from tensorflow import keras
import numpy as np

#Three numerical variables
num_data = np.random.random(size=(10,3))

#One categorical variables with 4 levels
cat_data = np.random.randint(0,4,10)

#Let's create one-hot encoded matrix since expected input_1 to have shape (4,)
one_hot_encoded_cat_data = np.eye(cat_data.max()+1)[cat_data]

target =  np.random.random(size=(10,1))

no_of_unique_cat  = len(np.unique(cat_data))
#Jeremy Howard provides the following rule of thumb; embedding size = min(50, number of categories/2).
embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
embedding_size = int(embedding_size)

# Use Input layers, specify input shape (dimensions except first)
inp_cat_data = keras.layers.Input(shape=(no_of_unique_cat,))
inp_num_data = keras.layers.Input(shape=(num_data.shape[1],))
# Bind nulti_hot to embedding layer
emb = keras.layers.Embedding(input_dim=no_of_unique_cat, output_dim=embedding_size)(inp_cat_data)  
# Also you need flatten embedded output of shape (?,3,2) to (?, 6) -
# otherwise it's not possible to concatenate it with inp_num_data
flatten = keras.layers.Flatten()(emb)
# Concatenate two layers
conc = keras.layers.Concatenate()([flatten, inp_num_data])
dense1 = keras.layers.Dense(3, activation=tf.nn.relu, )(conc)
# Creating output layer
out = keras.layers.Dense(1, activation=None)(dense1)
model = keras.Model(inputs=[inp_cat_data, inp_num_data], outputs=out)

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_squared_error])
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/model_summary_embedding.png?raw=true)

{% highlight python %}
model.fit([one_hot_encoded_cat_data, num_data], target)
# WARNING:tensorflow:From /anaconda3/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# 10/10 [==============================] - 0s 13ms/sample - loss: 0.1767 - mean_squared_error: 0.1767
# <tensorflow.python.keras.callbacks.History at 0xb2ff1efd0>
model.layers[1].get_weights()[0]
# array([[0.03832028, 0.01142023],
#        [0.0013773 , 0.05999473],
#        [0.04026476, 0.04118952],
#        [0.03986621, 0.0390432 ]], dtype=float32)
{% endhighlight %}
# REFERENCES
1. [https://towardsdatascience.com/decoded-entity-embeddings-of-categorical-variables-in-neural-networks-1d2468311635](https://towardsdatascience.com/decoded-entity-embeddings-of-categorical-variables-in-neural-networks-1d2468311635){:target="_blank"}
