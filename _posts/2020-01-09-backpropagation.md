---
layout: post
title: "A Step by Step Backpropagation Example for Regression using an One-hot Encoded Categorical Variable by hand and in Tensorflow"
author: "MMA"
comments: true
---

Backpropagation is a common method for training a neural network. It is nothing but a chain of rule. There is a lot of tutorials online, that attempt to explain how backpropagation works, but few that include an example with actual numbers. This post is my attempt to explain how it works with a concrete example using a regression example and a categorical variable which has been encoded using binary variables.

For this tutorial, we’re going to use a neural network with two numeric features and one categorical variable, three hidden neurons, one output neuron, since we are doing a regression analysis. We will have 4 observations. Each observations has two numeric features and one categorical variable. Therefore, design matrix is given by

$$
X= \begin{bmatrix}x_{11} & x_{12} & x_{13} \\  x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \\ x_{41} & x_{42} & x_{43}\end{bmatrix} = \begin{bmatrix}0.5 & 0.1 & 0 \\  0.3 & 0.2 & 1 \\ 0.7 & 0.9 & 2\\ 0.8 & 0.1 & 0\end{bmatrix}
$$

We one-hot encode the categorical variable.

$$
X = \begin{bmatrix}0.5 & 0.1 & 1 & 0 & 0 \\  0.3 & 0.2 & 0 & 1 & 0\\ 0.7 & 0.9 & 0 & 0 & 1\\ 0.8 & 0.1 & 1 & 0 & 0\end{bmatrix}
$$

We choose target variable to be:

$$
y = \begin{bmatrix}y_{1} \\ y_{2} \\ y_{3} \\ y_{4} \end{bmatrix} = \begin{bmatrix} 0.1 \\ 0.6 \\ 0.4 \\ 0.1 \end{bmatrix}
$$

Additionally, the hidden neurons and output neuron will include a bias. Biases for hidden layer neurons will be initialized from the same value. However, they can be different. Here’s the basic structure:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/nn_arch_backprop.png?raw=true)

We initialize the parameter as such:

$$
\begin{split}
\begin{bmatrix}\theta_{1} & \theta_{2} & \theta_{3}\\ \theta_{4} & \theta_{5} & \theta_{6}\\ \theta_{7} & \theta_{8}& \theta_{9}\\ \theta_{10} & \theta_{11}& \theta_{12}\\ \theta_{13} & \theta_{14}& \theta_{15}\end{bmatrix} &=\begin{bmatrix}0.19 & 0.55 & 0.76\\ 0.33 & 0.16 & 0.97 \\ 0.4 & 0.35 & 0.7\\ 0.51 & 0.85 & 0.85\\ 0.54 & 0.49 & 0.57\end{bmatrix}\\[3pt]
b_{1} = \begin{bmatrix}b_{1}^{1} \\ b_{1}^{2} \\ b_{1}^{3}\end{bmatrix} &= \begin{bmatrix} 0.1 \\ 0.1 \\ 0.1  \end{bmatrix}\\[3pt]
\begin{bmatrix}\theta_{16} \\ \theta_{17}\\ \theta_{18} \end{bmatrix} &= \begin{bmatrix} 0.1\\0.03\\-0.17 \end{bmatrix}\\[3pt]
b_{2} &= 0.1
\end{split}
$$

Any given path from an input neuron to an output neuron is essentially just a composition of functions; as such, we can use partial derivatives and the chain rule to define the relationship between any given weight and the cost function. We can use this knowledge to then leverage gradient descent algorithm in updating each of the weights. However, we will use a type of gradient descent which processes one training example per iteration, also known as Stochastic Gradient Descent (sometimes also referred to as iterative or on-line gradient descent). Hence, the parameters are being updated even after one iteration in which only a single example has been processed. In Stochastic Gradient Descent, we don't accumulate the weight updates. Instead, we update the weights after each training sample:

* for training sample i,
    - for each weight j
      $$
      \hat{\theta_{j}} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} J_{i}(\theta)
      $$
 
Here, $\alpha$ is learning rate and $J_{i}(\theta)$ is the cost for a single example $i$.

The term "stochastic" comes from the fact that the gradient based on a single training sample is a "stochastic approximation" of the "true" cost gradient. Due to its stochastic nature, the path towards the global cost minimum is not "direct" as in Gradient Descent, but may go "zig-zag" if we are visuallizing the cost surface in a 2D space. However, it has been shown that Stochastic Gradient Descent almost surely converges to the global cost minimum if the cost function is convex (or pseudo-convex) <a href="#note1" id="note1ref"><sup>1</sup></a>.

The cost function we try to minimize for a regression analysis is generally mean squared error which is given by:

$$
J(\theta) = \frac{1}{n} \sum_{i = 1}^{n} (\text{target}^{(i)} - \text{output}^{(i)})^2
$$

For a single example, the loss is:

$$
J_{i}(\theta) = (\text{target}^{(i)} - \text{output}^{(i)})^2
$$

Here, output is the result that is produced by the network for given inputs and target is the corresponding actual value for the observation $i$. 


## Observation 1

To begin, let's see what the neural network currently predicts given the weights and biases above and inputs of 0.5 and 0.1 and categorical variable 'Category0'. In order to do this we will feed those inputs forward though the network.

### The Forward Pass 

We first figure out the total net input to each hidden layer neuron, squash the total net input using an activation function (here we use the sigmoid function), then repeat the process with the output layer neurons.

Here’s how we calculate the total net input for $h_1$ using the first observation:

$$
\begin{split}
net_{h_{1}} &= input_{1} * \theta_{1} +  input_{2} * \theta_{4} + \text{DummyCat0} * \theta_{7} + \text{DummyCat1} * \theta_{10} + \text{DummyCat2} * \theta_{13} + 1 * b_{1}\\ 
&= 0.5*0.19 + 0.1*0.33 + 1*0.4 + 0*0.51 + 0*0.54 + 1*0.1 \\
&=0.628
\end{split}
$$

We then squash it using the sigmoid function to get the output of $h_1$:

$$
out_{h_{1}} = \frac{1}{1+e^{-net_{h_{1}}}} = \frac{1}{1+e^{- 0.628}} = 0.65203583
$$

Carrying out the same process for $h_2$ and $h_{3}$, we get:

$$
\begin{split}
net_{h_{2}} &= input_{1} * \theta_{2} +  input_{2} * \theta_{5} + \text{DummyCat0} * \theta_{8} + \text{DummyCat1} * \theta_{11} + \text{DummyCat2} * \theta_{14} + 1 * b_{1}\\ 
&= 0.5*0.55 + 0.1*0.16 + 1*0.35 + 0*0.85 + 0*0.49 + 1*0.1 \\
&=0.741
\end{split}
$$

$$
out_{h_{2}} = \frac{1}{1+e^{-net_{h_{1}}}} = \frac{1}{1+e^{- 0.741}} = 0.6772145
$$

$$
\begin{split}
net_{h_{3}} &= input_{1} * \theta_{3} +  input_{2} * \theta_{6} + \text{DummyCat0} * \theta_{9} + \text{DummyCat1} * \theta_{12} + \text{DummyCat2} * \theta_{15} +  1 * b_{1}\\ 
&= 0.5*0.76 + 0.1*0.97 + 1*0.7 + 0*0.85 + 0*0.57 + 1*0.1 \\
&=1.277
\end{split}
$$

$$
out_{h_{3}}  = \frac{1}{1+e^{-net_{h_{1}}}} = \frac{1}{1+e^{- 0.1.277}} = 0.78193873
$$

We repeat this process for the output layer neuron, using the output from neurons of the hidden layer as inputs. Here, since we have regression, we will have one neuron in the output layer. We do not have any activation function for this layer. So, we do not need to pass it through to squash the net value. Here’s the output for $output$

$$
\begin{split}
\text{output}^{(1)} &= out_{h_{1}} * \theta_{16} + out_{h_{2}} * \theta_{17} + out_{h_{3}} * \theta_{18} + 1*b_{2}\\
&= 0.65203583 * 0.1 + 0.6772145 * 0.03 + 0.78193873 * (-0.17) + 1*0.1 \\
&= 0.05259044
\end{split}
$$

#### Calculating the Total Error

We can now calculate the error for the output neuron using the squared error function. The target output for observation 1 is $0.1$ but the neural network output $0.05259044$, therefore its error is:

$$
\begin{split}
J_{1}(\theta) &= (\text{target}^{(1)} - \text{output}^{(1)})^{2}\\
&= (0.1 - 0.05259044)^{2}\\
&=0.002247666008770466
\end{split}
$$

This is our loss value for observation 1. Since this is not zero, we can say that there is still room for improvement the model.

### The Backward Pass

Our goal with backpropagation is to update each of the weights in the network so that the actual output might be closer the target output, thereby minimizing the error for output neuron and the network as a whole.

#### From Output Layer Neuron to Bias of Output Layer

Let's first consider $b_{2}$. We want to know how much a change in $b_{2}$ affects the total error. In other words, we would like to compute $\frac{\partial J_{1}(\theta)}{\partial b_{2}}$.

By applying the chain rule we know that:

$$
\frac{\partial J_{1}(\theta)}{\partial b_{2}} = \frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial b_{2}}
$$

We need to figure out each piece in this equation. First, we need to find how much the loss changes with respect to the output for observation 1?

$$
\begin{split}
J_{1}(\theta) &= (\text{target}^{(1)} - \text{output}^{(1)})^{2}\\
\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} &= -2 * (\text{target}^{(1)} - \text{output}^{(1)})\\
&= -2 * (0.1 - 0.05259044)\\
&=-0.09481912
\end{split}
$$

Next, we need to compute how much the $\text{output}^{(1)}$ change with respect to bias for the neuron of output layer.

$$
\begin{split}
\text{output}^{(1)} &= out_{h_{1}} * \theta_{16} + out_{h_{2}} * \theta_{17} + out_{h_{3}} * \theta_{18} + 1 * b_{2}\\
\frac{\partial \text{output}^{(1)}}{\partial b_{2}} &= 1
\end{split}
$$

By putting it all together, we will have

$$
\begin{split}
\frac{\partial J_{1}(\theta)}{\partial b_{2}} &= \frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial b_{2}} \\
&=-0.09481912 * 1\\
&= - 0.09481912
\end{split}
$$

To decrease the loss, we then subtract this value from the current weight, multiplied by some learning rate $\alpha$:

$$
b_{2}^{+} = b_{2} - \alpha * \frac{\partial J_{1}(\theta)}{\partial b_{2}} = 0.1 - 0.5 * (-0.09481912) = 0.14740956
$$

#### From Output Layer to Hidden Layer

Let's now consider $\theta_{16}$. We want to know how much a change in $\theta_{16}$ affects the total error. In other words, we would like to compute $\frac{\partial J_{1}(\theta)}{\partial \theta_{16}}$. 

By applying the chain rule, again, we know that:

$$
\frac{\partial J_{1}(\theta)}{\partial \theta_{16}} = \frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial \theta_{16}}
$$

We need to figure out each piece in this equation. We already know how much the loss changes with respect to the output for observation 1.

$$
\begin{split}
J_{1}(\theta) &= (\text{target}^{(1)} - \text{output}^{(1)})^{2}\\
\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} &= -2 * (\text{target}^{(1)} - \text{output}^{(1)})\\
&= -2 * (0.1 - 0.05259044)\\
&=-0.09481912
\end{split}
$$

Next, we need to compute how much the $\text{output}^{(1)}$ change with respect to $\theta_{16}$.

$$
\begin{split}
\text{output}^{(1)} &= out_{h_{1}} * \theta_{16} + out_{h_{2}} * \theta_{17} + out_{h_{3}} * \theta_{18} + 1 * b_{2}\\
\frac{\partial \text{output}^{(1)}}{\partial \theta_{16}} &= out_{h_{1}} = 0.65203583
\end{split}
$$

By putting it all together, we will have

$$
\begin{split}
\frac{\partial J_{1}(\theta)}{\partial \theta_{16}} = \frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial \theta_{16}}
&=-0.09481912 *  0.65203583\\
&= - 0.061825463
\end{split}
$$

To decrease the loss, we then subtract this value from the current weight, multiplied by some learning rate $\alpha$:

$$
\theta_{16}^{+} = \theta_{16} - \alpha * \frac{\partial J_{1}(\theta)}{\partial \theta_{16}} = 0.1 - 0.5 * (-0.061825463) = 0.1309127315
$$

We will repeat the same process for $\theta_{17}$ and $\theta_{18}$.

For $\theta_{17}$:

$$
\begin{split}
\theta_{17}^{+} &= \theta_{17} - \alpha * \frac{\partial J_{1}(\theta)}{\partial \theta_{17}} \\
&= \theta_{17} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial \theta_{17}} \right) \\
&= \theta_{17} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * out_{h_{2}} \right) \\
&= 0.03 - 0.5 * (-0.09481912 * 0.6772145)\\
&= 0.03 - 0.5 * (-0.06421288)\\
&= 0.06210644
\end{split}
$$

For $\theta_{18}$:

$$
\begin{split}
\theta_{18}^{+} &= \theta_{18} - \alpha * \frac{\partial J_{1}(\theta)}{\partial \theta_{18}} \\
&= \theta_{18} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial \theta_{18}} \right) \\
&= \theta_{18} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * out_{h_{3}} \right) \\
&= -0.17 - 0.5 * (-0.09481912 * 0.78193873)\\
&= -0.17 - 0.5 * (-0.07414274)\\
&= -0.13292864
\end{split}
$$

#### From Output Layer Neuron to Bias of Hidden Layer

Computing $b_{1}$ is a bit trickier because each element of this vector gets contribution from different neurons. All we need to do is take account whichever neuron is feeding whichever bias. 

Let's start with $b_{1}^{1}$. This offset value takes its contribution from the first neuron of the hidden layer, i.e., $h_{1}$:

$$
\begin{split}
b_{1}^{1+} &= b_{1}^{1} - \alpha * \frac{\partial J_{1}(\theta)}{\partial b_{1}^{1}} \\
&= b_{1}^{1} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial out_{h_{1}}} * \frac{\partial out_{h_{1}}}{\partial net_{h_{1}}} * \frac{\partial net_{h_{1}}}{\partial b_{1}^{1}}\right) \\
&=0.1 - 0.5 * (-0.09481912 * 0.1 *  0.65203583 * (1- 0.65203583) * 1)\\
&=0.1 - 0.5 * (-0.0021513)\\
&=0.10107566
\end{split}
$$

There are couple of things different we have while we compute for $b_{1}^{1}$ as you can realize easily:

Since we know

$$
\text{output}^{(1)} = out_{h_{1}} * \theta_{16} + out_{h_{2}} * \theta_{17} + out_{h_{3}} * \theta_{18} + 1*b_{2}
$$

Therefore, 

$$
\frac{\partial \text{output}^{(1)}}{\partial out_{h_{1}}} = \theta_{16}
$$ 

Similarly, since we know, 

$$
out_{h_{1}} = Sigmoid(net_{h_{1}}) = \frac{1}{1+e^{-net_{h_{1}}}}
$$

Therefore,

$$
\frac{\partial out_{h_{1}}}{\partial net_{h_{1}}} = Sigmoid(net_{h_{1}}) * (1-Sigmoid(net_{h_{1}}))
$$

and

since we know

$$
net_{h_{1}} = input_{1} * \theta_{1} +  input_{2} * \theta_{4} + \text{DummyCat0} * \theta_{7} + \text{DummyCat1} * \theta_{10} + \text{DummyCat2} * \theta_{13} + 1 * b_{1}
$$

Therefore, we would get 

$$
\frac{\partial net_{h_{1}}}{\partial b_{1}^{1}} = 1
$$

If we repeat the same process for the second element of bias vector, $b_{1}^{2}$ and third element $b_{1}^{3}$, we will get:

$$
\begin{split}
b_{1}^{2+} &= b_{1}^{2} - \alpha * \frac{\partial J_{1}(\theta)}{\partial b_{1}^{2}} \\
&= b_{1}^{2} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial out_{h_{2}}} * \frac{\partial out_{h_{2}}}{\partial net_{h_{2}}} * \frac{\partial net_{h_{2}}}{\partial b_{1}^{2}}\right) \\
&=0.1 - 0.5 * (-0.09481912 * 0.03 *  0.6772145 * (1- 0.6772145) * 1)\\
&=0.1 - 0.5 * (-0.00062181)\\
&=0.10031091
\end{split}
$$

$$
\begin{split}
b_{1}^{3+} &= b_{1}^{3} - \alpha * \frac{\partial J_{1}(\theta)}{\partial b_{1}^{3}} \\
&= b_{1}^{3} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial out_{h_{3}}} * \frac{\partial out_{h_{3}}}{\partial net_{h_{3}}} * \frac{\partial net_{h_{3}}}{\partial b_{1}^{3}}\right) \\
&=0.1 - 0.5 * (-0.09481912 * -0.17 *  0.78193873 * (1- 0.78193873) * 1)\\
&=0.1 - 0.5 * (0.0027485)\\
&=0.09862575
\end{split}
$$

#### From Output Layer to Input Layer

All the same logic applies for the weights, $\theta_{1}, \theta_{2}, \dots, \theta_{15}$. We will show the process for one weight and the rest can be easily done by you!

$\theta_{1}$ gets the contribution only from $h_{1}$.

$$
\begin{split}
\theta_{1}^{+} &= \theta_{1} - \alpha * \frac{\partial J_{1}(\theta)}{\partial \theta_{1}} \\
&= \theta_{1} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial out_{h_{1}}} * \frac{\partial out_{h_{1}}}{\partial net_{h_{1}}} * \frac{\partial net_{h_{1}}}{\partial \theta_{1}}\right) \\
&=0.19 - 0.5 * (-0.09481912 * 0.1 *   0.65203583 * (1- 0.65203583) * 0.5)\\
&=0.19 - 0.5 * (-0.0010756523064)\\
&=0.19053783
\end{split}
$$

Here, 

$$
\frac{\partial net_{h_{1}}}{\partial \theta_{1}} = input_{1} = 0.5
$$

because

$$
net_{h_{1}} = input_{1} * \theta_{1} +  input_{2} * \theta_{4} + \text{DummyCat0} * \theta_{7} + \text{DummyCat1} * \theta_{10} + \text{DummyCat2} * \theta_{13} + 1 * b_{1}
$$


The same applies for $\theta_{2}, \theta_{3}, \theta_{4}, \theta_{5}, \theta_{6}$. For now let's skip those and do the computations for neuron representing $\text{DummyCat0}$ and $\text{DummyCat1}$ to see how one hot encoding affects the gradient updates. Therefore, let's do the computations for $\theta_{7}$ and $\theta{10}$.

For $\theta_{7}$, the weight will be updated as such:

$$
\begin{split}
\theta_{7}^{+} &= \theta_{7} - \alpha * \frac{\partial J_{1}(\theta)}{\partial \theta_{7}} \\
&= \theta_{7} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial out_{h_{1}}} * \frac{\partial out_{h_{1}}}{\partial net_{h_{1}}} * \frac{\partial net_{h_{1}}}{\partial \theta_{7}}\right) \\
&=0.4 - 0.5 * (-0.09481912 * 0.1 *   0.2268851063962111 * 1)\\
&=0.4 - 0.5 * (-0.002151304612959511)\\
&=0.40107566
\end{split}
$$

by using the facts that are:

$$
\frac{\partial \text{output}^{(1)}}{\partial out_{h_{1}}} = \theta_{16} = 0.1
$$

and

$$
\frac{\partial out_{h_{1}}}{\partial net_{h_{1}}} = Sigmoid(net_{h_{1}}) * (1-Sigmoid(net_{h_{1}})) =  0.65203583 * (1- 0.65203583) = 0.2268851063962111
$$

and

$$
\frac{\partial net_{h_{1}}}{\partial \theta_{7}} = \text{DummyCat0} = 1
$$

because

$$
net_{h_{1}} = input_{1} * \theta_{1} +  input_{2} * \theta_{4} + \text{DummyCat0} * \theta_{7} + \text{DummyCat1} * \theta_{10} + \text{DummyCat2} * \theta_{13} + 1 * b_{1}
$$

For $\theta_{10}$, the calculations are the same. But this time, because of the fact that we will have gradient of $net_{h_{1}}$ with respect to $\theta_{10}$ as zero, therefore, there will be no gradient update for this particular weight.

$$
\begin{split}
\theta_{10}^{+} &= \theta_{10} - \alpha * \frac{\partial J_{1}(\theta)}{\partial \theta_{10}} \\
&= \theta_{10} - \alpha * \left(\frac{\partial J_{1}(\theta)}{\partial \text{output}^{(1)}} * \frac{\partial \text{output}^{(1)}}{\partial out_{h_{1}}} * \frac{\partial out_{h_{1}}}{\partial net_{h_{1}}} * \frac{\partial net_{h_{1}}}{\partial \theta_{10}}\right) \\
&=0.51 - 0.5 * (-0.09481912 * 0.1 *   0.2268851063962111 * 0)\\
&=0.51- 0.5 * (0)\\
&=0.51 
\end{split}
$$

we will leave it as it is here since the rest of the computations are fairly simple and straightforward. You have to repeat the process for all other weights and continue to do so for all observations. After the last update (of observation 4), loss will be 2.02787184715271 and weight are:

$$
\begin{split}
\begin{bmatrix}\theta_{1} & \theta_{2} & \theta_{3}\\ \theta_{4} & \theta_{5} & \theta_{6}\\ \theta_{7} & \theta_{8}& \theta_{9}\\ \theta_{10} & \theta_{11}& \theta_{12}\\ \theta_{13} & \theta_{14}& \theta_{15}\end{bmatrix} &=\begin{bmatrix}0.06418779& 0.42243242& 0.6311462 \\ 0.25548685 & 0.08999705 & 0.9365066 \\0.30406353 & 0.24792391 & 0.55890244 \\ 0.52213347 & 0.854805 & 0.84091777 \\ 0.46517092 & 0.42249298 & 0.5504809 \end{bmatrix}\\[3pt]
b_{1} = \begin{bmatrix}b_{1}^{1} \\ b_{1}^{2}\\ b_{1}^{3}\end{bmatrix} &= \begin{bmatrix}-0.05863208\\-0.06477813\\-0.06969891  \end{bmatrix}\\[3pt]
\begin{bmatrix}\theta_{16} \\ \theta_{17}\\ \theta_{18} \end{bmatrix} &= \begin{bmatrix}0.6166916\\0.6466775\\0.514491  \end{bmatrix}\\[3pt]
b_{2} &= 1.0416498
\end{split}
$$

# Tensorflow Implementation

If you implement exact same problem in Tensorflow, you will get the results:

{% highlight python %}
#Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
tf.set_random_seed(seed=42)

import math

num_data = np.array([[0.5, 0.1], [0.3, 0.2], [0.7, 0.9],[0.8, 0.1]])
#(4, 2)
# array([[0.5, 0.1],
#        [0.3, 0.2],
#        [0.7, 0.9],
#        [0.8, 0.1]])

cat_data = np.array([[0], [1], [2], [0]])
#(4, 1)

one_hot_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
#(4, 3)

target = np.array([[0.1], [0.6], [0.4], [0.1]])
# (4, 1)

def get_trainable_variables(graph=tf.get_default_graph()):
    return [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

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

data = np.concatenate((num_data, one_hot_cat), axis = 1)
#(4, 5)

n_features = data.shape[1]
n_outputs = target.shape[1]
n_hidden = 3

tf.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('Placeholders'):
        X = tf.placeholder('float', shape=[None, n_features])
        #<tf.Tensor 'Placeholder:0' shape=(?, 5) dtype=float32>
        y = tf.placeholder('float', shape=[None, n_outputs])
        #<tf.Tensor 'Placeholder_1:0' shape=(?, 1) dtype=float32>

    with tf.name_scope("First_Layer"):
        W_fc1 = tf.get_variable('First_Layer/Hidden_layer_weights', initializer=tf.constant(np.array([[0.19, 0.55, 0.76],[0.33, 0.16, 0.97],[0.4 , 0.35, 0.7 ],[0.51, 0.85, 0.85],[0.54, 0.49, 0.57]]), dtype=tf.float32))
        #<tf.Variable 'First_Layer/Variable:0' shape=(5, 3) dtype=float32_ref>
        b_fc1 = tf.get_variable('First_Layer/Biases', initializer=tf.constant(np.array([0.1, 0.1, 0.1]), dtype=tf.float32))
        #<tf.Variable 'First_Layer/Variable_1:0' shape=(3,) dtype=float32_ref>
        h_fc1 = tf.nn.sigmoid(tf.matmul(X, W_fc1) + b_fc1)
        #<tf.Tensor 'First_Layer/Relu:0' shape=(?, 3) dtype=float32>

    with tf.name_scope("Output_Layer"):
        W_fc2 = tf.get_variable('Output_Layer/Output_layer_weights', initializer=tf.constant(np.array([[ 0.10],[ 0.03],[-0.17]]), dtype=tf.float32))
        # <tf.Variable 'Output_Layer/Variable:0' shape=(3, 1) dtype=float32_ref>
        b_fc2 = tf.get_variable('Output_Layer/Biases', initializer=tf.constant(np.array([0.1]), dtype=tf.float32))
        # <tf.Variable 'Output_Layer/Variable_1:0' shape=(1,) dtype=float32_ref>
        y_pred = tf.cast(tf.matmul(h_fc1, W_fc2) + b_fc2, dtype = tf.float32)
        #<tf.Tensor 'Output_Layer/add:0' shape=(?, 1) dtype=float32>

    with tf.name_scope("Loss"):
        loss = tf.losses.mean_squared_error(labels = y, predictions = y_pred)

    with tf.name_scope('Train'):
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        grads_and_vars = optimizer.compute_gradients(loss)
        trainer = optimizer.apply_gradients(grads_and_vars)

    # [(<tf.Tensor 'Train/gradients/First_Layer/MatMul_grad/tuple/control_dependency_1:0' shape=(5, 3) dtype=float32>,
    #   <tf.Variable 'First_Layer/Variable:0' shape=(5, 3) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/First_Layer/add_grad/tuple/control_dependency_1:0' shape=(3,) dtype=float32>,
    #   <tf.Variable 'First_Layer/Variable_1:0' shape=(3,) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/Output_Layer/MatMul_grad/tuple/control_dependency_1:0' shape=(3, 1) dtype=float32>,
    #   <tf.Variable 'Output_Layer/Variable:0' shape=(3, 1) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/Output_Layer/add_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>,
    #   <tf.Variable 'Output_Layer/Variable_1:0' shape=(1,) dtype=float32_ref>)]

    with tf.name_scope("Init"):
        global_variables_init = tf.global_variables_initializer()
        
get_trainable_variables(graph=graph)
# [<tf.Variable 'First_Layer/Hidden_layer_weights:0' shape=(5, 3) dtype=float32_ref>,
#  <tf.Variable 'First_Layer/Biases:0' shape=(3,) dtype=float32_ref>,
#  <tf.Variable 'Output_Layer/Output_layer_weights:0' shape=(3, 1) dtype=float32_ref>,
#  <tf.Variable 'Output_Layer/Biases:0' shape=(1,) dtype=float32_ref>]

with tf.Session(graph=graph) as sess:
    global_variables_init.run()
    tf.get_default_graph().finalize()
    print("Initialized")
    
    print ("Variables before training")
    old_var = {}
    for var in tf.global_variables():
        old_var[var.name] = sess.run(var)
        #print (var.name, sess.run(var))
    print(old_var)
    print('\n\n')
    
    miniBatches = miniBatch(data, target, batchSize = 1)
    total_batch = len(miniBatches) 
    i=1
    for batch in miniBatches:
        print('\n{}-observation\n'.format(i))
        xBatch = batch[0]
        yBatch = batch[1]
        _, loss_val, h_fc1_val, grads_and_vars_val, y_pred_val = sess.run([trainer, loss, h_fc1, grads_and_vars, y_pred], feed_dict={X: xBatch, y: yBatch})
        print('Loss: {}'.format(loss_val))
        print('Prediction: {}'.format(y_pred_val))
        print('Hidden layer forward prop:{}'.format(h_fc1_val))
        print('\n\n')
        print(grads_and_vars_val)
        i += 1
    print("Optimization Finished!")   
    print('\n\n')
    print ("Variables after training")
    new_var = {}
    for var in tf.global_variables():
        new_var[var.name] = sess.run(var)
    print(new_var)
        
# Initialized
# Variables before training
# {'First_Layer/Hidden_layer_weights:0': array([[0.19, 0.55, 0.76],
#        [0.33, 0.16, 0.97],
#        [0.4 , 0.35, 0.7 ],
#        [0.51, 0.85, 0.85],
#        [0.54, 0.49, 0.57]], dtype=float32), 'First_Layer/Biases:0': array([0.1, 0.1, 0.1], dtype=float32), 'Output_Layer/Output_layer_weights:0': array([[ 0.1 ],
#        [ 0.03],
#        [-0.17]], dtype=float32), 'Output_Layer/Biases:0': array([0.1], dtype=float32)}




# 1-observation

# Loss: 0.002247666008770466
# Prediction: [[0.05259044]]
# Hidden layer forward prop:[[0.65203583 0.6772145  0.78193873]]



# [(array([[-1.0756523e-03, -3.1090478e-04,  1.3742511e-03],
#        [-2.1513047e-04, -6.2180960e-05,  2.7485023e-04],
#        [-2.1513046e-03, -6.2180957e-04,  2.7485022e-03],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=float32), array([[0.19053783, 0.55015546, 0.75931287],
#        [0.33010757, 0.16003108, 0.9698626 ],
#        [0.40107566, 0.3503109 , 0.69862574],
#        [0.51      , 0.85      , 0.85      ],
#        [0.54      , 0.49      , 0.57      ]], dtype=float32)), (array([-0.0021513 , -0.00062181,  0.0027485 ], dtype=float32), array([0.10107566, 0.10031091, 0.09862575], dtype=float32)), (array([[-0.06182546],
#        [-0.06421288],
#        [-0.07414274]], dtype=float32), array([[ 0.13091274],
#        [ 0.06210644],
#        [-0.13292864]], dtype=float32)), (array([-0.09481911], dtype=float32), array([0.14740956], dtype=float32))]

# 2-observation

# Loss: 0.17892064154148102
# Prediction: [[0.17700991]]
# Hidden layer forward prop:[[0.67573905 0.7590291  0.7974435 ]]



# [(array([[-0.0072801 , -0.00288298,  0.00544937],
#        [-0.0048534 , -0.00192198,  0.00363291],
#        [ 0.        ,  0.        ,  0.        ],
#        [-0.02426698, -0.00960992,  0.01816456],
#        [ 0.        ,  0.        ,  0.        ]], dtype=float32), array([[0.19417787, 0.55159694, 0.75658816],
#        [0.33253425, 0.16099207, 0.9680461 ],
#        [0.40107566, 0.3503109 , 0.69862574],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.54      , 0.49      , 0.57      ]], dtype=float32)), (array([-0.02426698, -0.00960992,  0.01816456], dtype=float32), array([0.11320915, 0.10511587, 0.08954347], dtype=float32)), (array([[-0.5716619 ],
#        [-0.6421236 ],
#        [-0.67462146]], dtype=float32), array([[0.4167437 ],
#        [0.38316822],
#        [0.20438209]], dtype=float32)), (array([-0.8459802], dtype=float32), array([0.57039964], dtype=float32))]

# 3-observation

# Loss: 0.907796323299408
# Prediction: [[1.3527834]]
# Hidden layer forward prop:[[0.748083   0.7551234  0.88699394]]



# [(array([[0.10476074, 0.09450983, 0.02732672],
#        [0.13469239, 0.12151264, 0.03513435],
#        [0.        , 0.        , 0.        ],
#        [0.        , 0.        , 0.        ],
#        [0.1496582 , 0.13501404, 0.03903817]], dtype=float32), array([[0.1417975 , 0.504342  , 0.7429248 ],
#        [0.26518807, 0.10023575, 0.950479  ],
#        [0.40107566, 0.3503109 , 0.69862574],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.46517092, 0.42249298, 0.5504809 ]], dtype=float32)), (array([0.1496582 , 0.13501404, 0.03903817], dtype=float32), array([0.03838005, 0.03760885, 0.07002439], dtype=float32)), (array([[1.4255222],
#        [1.4389381],
#        [1.6902263]], dtype=float32), array([[-0.2960174 ],
#        [-0.33630085],
#        [-0.6407311 ]], dtype=float32)), (array([1.9055669], dtype=float32), array([-0.38238382], dtype=float32))]

# 4-observation

# Loss: 2.02787184715271
# Prediction: [[-1.3240336]]
# Hidden layer forward prop:[[0.6409322  0.69027746 0.8112324 ]]



# [(array([[0.15521942, 0.16381918, 0.22355728],
#        [0.01940243, 0.0204774 , 0.02794466],
#        [0.19402426, 0.20477396, 0.2794466 ],
#        [0.        , 0.        , 0.        ],
#        [0.        , 0.        , 0.        ]], dtype=float32), array([[0.06418779, 0.42243242, 0.6311462 ],
#        [0.25548685, 0.08999705, 0.9365066 ],
#        [0.30406353, 0.24792391, 0.55890244],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.46517092, 0.42249298, 0.5504809 ]], dtype=float32)), (array([0.19402426, 0.20477396, 0.2794466 ], dtype=float32), array([-0.05863208, -0.06477813, -0.06969891], dtype=float32)), (array([[-1.825418 ],
#        [-1.9659567],
#        [-2.3104444]], dtype=float32), array([[0.6166916],
#        [0.6466775],
#        [0.5144911]], dtype=float32)), (array([-2.8480673], dtype=float32), array([1.0416498], dtype=float32))]
# Optimization Finished!



# Variables after training
# {'First_Layer/Hidden_layer_weights:0': array([[0.06418779, 0.42243242, 0.6311462 ],
#        [0.25548685, 0.08999705, 0.9365066 ],
#        [0.30406353, 0.24792391, 0.55890244],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.46517092, 0.42249298, 0.5504809 ]], dtype=float32), 'First_Layer/Biases:0': array([-0.05863208, -0.06477813, -0.06969891], dtype=float32), 'Output_Layer/Output_layer_weights:0': array([[0.6166916],
#        [0.6466775],
#        [0.5144911]], dtype=float32), 'Output_Layer/Biases:0': array([1.0416498], dtype=float32)}
{% endhighlight %}

<a id="note1" href="#note1ref"><sup>1</sup></a>Bottou, Léon (1998). "Online Algorithms and Stochastic Approximations". Online Learning and Neural Networks. Cambridge University Press. ISBN 978-0-521-65263-6.
