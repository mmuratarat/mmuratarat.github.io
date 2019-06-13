---
layout: post
title: "How to use embedding layer and other feature columns together in a network using Keras?"
author: "MMA"
comments: true
---

# Why should you use an embedding layer? 

Here are the two main reasons:
 
One-Hot encoding is a commonly used method for converting a categorical input variable into continuous variable. For every level present, one new variable will be created.Presence of a level is represent by 1 and absence is represented by 0. However, one-hot encoded vectors are high-dimensional and sparse. One-hot encoding of high cardinality features often results in an unrealistic amount of computational resource requirement. It treats different values of categorical variables completely independent of each other and often ignores the informative relations between them.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/dummying.png?raw=true)

The vectors of each embedding are learned while training the neural network.The embedding reduces the memory usage and speeds up the training comparing with one-hot encoding. More importantly though, this approach allows for relationships between categories to be captured. Perhaps Saturday and Sunday have similar behavior, and maybe Friday behaves like an average of a weekend and a weekday.
 
For instance, a 4-dimensional version of an embedding for day of week could look like:

| Days    	| Embeddings       	|
|---------	|------------------	|
| Sunday  	| [.8, .2, .1, .1] 	|
| Monday  	| [.1, .2, .9, .9] 	|
| Tuesday 	| [.2, .1, .9, .8] 	|

Here, Monday and Tuesday are fairly similar, yet they are both quite different from Sunday. Again, this is a toy example.

The advantage of doing this compared to the traditional approach of creating dummy variables (i.e. doing one hot encodings), is that each day can be represented by four numbers instead of one, hence we gain higher dimensionality and much richer relationships. 

Another advantage of embeddings is that the learned embeddings can be visualized to show which categories are similar to each other. The most popular method for this is t-SNE, which is a technique for dimensionality reduction that works particularly well for visualizing data sets with high-dimensionality.
 
# Embedding Dimensionality
The embedding-size defines the dimensionality in which we map the categorical variables. Jeremy Howard provides a general rule of thumb about the number of embedding dimensions: embedding size = min(50, number of categories/2). This [Google Blog](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html) also tells that a good rule of thumb is 4th root of the number of categories. Therefore, So it’s kind of experimental. However, literature shows that embedding dimensions of size 50 produces the most accurate results.

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

# REFERENCES
1. [https://towardsdatascience.com/decoded-entity-embeddings-of-categorical-variables-in-neural-networks-1d2468311635](https://towardsdatascience.com/decoded-entity-embeddings-of-categorical-variables-in-neural-networks-1d2468311635){:target="_blank"}
