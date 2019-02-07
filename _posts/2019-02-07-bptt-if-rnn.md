---
layout: post
title: "Backpropagation Through Time for Recurrent Neural Network"
author: "MMA"
comments: true
---

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/BPTT.png)

The dynamical system of a Recurrent Neural Network is defined by:

$$
\begin{split}
    h_{t} & = f_{h} (X_{t}, h_{t-1})\\
    y_{t} &= f_{o}(h_{t})
\end{split}
$$


A conventional RNN is constructed by defining the transition function and the output function for a single instance:

$$
\begin{split}
    h_{t} & = f_{h} (X_{t}, h_{t-1}) = tanh(W_{xh}^{T} \cdot X_{t} + W_{hh}^{T}\cdot h_{t-1} +b_{h})\\
    o_{t} &= W_{yh}^{T}\cdot h_{t} + b_{y}\\
    \hat{y}_{t} &= f_{o}(h_{t}) = softmax(o_{t})
\end{split}
$$

where 

1. $h_{t}$: Hidden state at the time-step $t$.
2. $h_{t-1}$: Hidden state at the timne step $t-1$.
3. $W_{xh}$ Weight matrix for input to hidden layer.
4. $W_{hh}$ Weight matrix for recurrent connections.
5. tanh: Hyperbolic tangent activation function
6. $X_{t}$ Input at the time-step $t$.

In order to do backpropagation through time to train an RNN, we need to compute the loss function first:

$$
L (\hat{y}, y) = \sum_{t = 1}^{T} L(\hat{y}_{t}, y_{t}) = -\sum_{t = 1}^{T}y_{t} log \hat{y}_{t}
$$

We have $o_{t} = W_{yh}^{T}\cdot h_{t} + b_{y}$, then $\hat{y}_{t} = softmax(o_{t})$.


$$
L (\hat{y}, y) = -\sum_{t = 1}^{T}y_{t} log \left[softmax(o_{t}) \right]
$$


By taking the derivative of $L$ with respect to $o_{t}$, we get the following:

$$
\frac{\partial}{\partial o_{t}} L (\hat{y}, y) = -(y_{t} - \hat{y}_{t})
$$

Note that the weight $W_{yh}$ is shared across all the time sequence. Therefore, we can differentiate to it at the each time step and summ all together:

$$
\frac{\partial L (\hat{y}, y)}{\partial W_{yh}}  = \sum_{t = 1}^{T} \frac{\partial L}{\partial \hat{y}_{t}} \frac{\partial \hat{y}_{t}}{\partial W_{yh}}
$$

Similarly, we can get the gradient w.r.t. bias $b_{y}$:

$$
\frac{\partial L (\hat{y}, y)}{\partial b_{y}}  = \sum_{t = 1}^{T} \frac{\partial L}{\partial \hat{y}_{t}} \frac{\partial \hat{y}_{t}}{\partial b_{y}}
$$

Further, let's use $L_{t+1}$ to denote the output of the time-step $t+1$, $L_{t+1} = -y_{t+1} log \hat{y}_{t+1}$.

Now, let's go throught the details to derive the gradient with respect to $W_{hh}$, considering at the time step $t \rightarrow t+1$ (from time-step $t$ to $t+1$).

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} =  \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial W_{hh}}
$$

where we consider only one time-step ($t \rightarrow t+1$). But, because the hidden state $h_{t+1}$ partially depends also on $h_{t}$, we can use backpropagation through time to compute the partial derivative above. Thinking further, $W_{hh}$ is shared across the whole time sequence, according to the recursive formulation ($h_{t} = tanh(W_{xh}^{T} \cdot X_{t} + W_{hh}^{T}\cdot h_{t-1} +b_{h})$). Thus, at the time-step $t-1 \rightarrow t$, we can further get the partial derivative with respect to $W_{hh}$ as the following:

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} =  \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{t}} \frac{\partial h_{t}}{\partial W_{hh}}
$$

Thus, at the time-step $t+1$, we can compute the gradient with respect to $\hat{y}_{t+1}$ and further use backpropagation through time from t to 1 to compute the gradient with respect to $W_{hh}$. Therefore, if we only consider the output $y_{t+1}$ at the time-step $t+1$

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} = \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W_{hh}}
$$

Note that $\frac{\partial h_{t+1}}{\partial h_{k}}$ is a chain rule in itself!  For example, $\frac{\partial h_{3}}{\partial h_{1}} = \frac{\partial h_{3}}{\partial h_{2}}\frac{\partial h_{2}}{\partial h_{1}}$. Also note that because we are taking the derivative of a vector function with respect to a vector, the result is a matrix (called the Jacobian matrix) whose elements are all the pointwise derivatives. We can rewrite the above gradient:

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} = \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}  \left( \prod_{j = k} ^{t} \frac{\partial h_{j+1}}{\partial h_{j}} \right) \frac{\partial h_{k}}{\partial W_{hh}}
$$

Aggregate thge gradients with respect to $W_{hh}$ over the whote time-sequence with backpropagation, we can finally yield the following gradient with respect to $W_{hh}$:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t}^{T} \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_{k}}\frac{\partial h_{k}}{\partial W_{hh}}
$$

Now, let's work on to derive the gradient with respect to $W_{xh}$. Similarly, we consider the time-step $t+1$ (which gets only contribution from $X_{t+1}$) and calculate the gradients with respect to $W_{xh}$ as follows:

$$
\frac{\partial L_{t+1}}{\partial W_{xh}} = \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial W_{xh}} 
$$

Because $h_{t}$ and $X_{t+1}$ both make contribution to $h_{t+1}$, we need to backpropagate to $h_{t}$ as well.

If we consider the contribution from the time-step, we can further get:

$$
\frac{\partial L_{t+1}}{\partial W_{xh}} = \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial W_{xh}}  + \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial  h_{t}} \frac{\partial h_{t}}{\partial W_{xh}} 
$$

Thus, summing up all the contributions from $t+1$ to $1$ via Backpropagation, we can yield the gradient at the time-step $t+1$:

$$
\frac{\partial L_{t+1}}{\partial W_{xh}} = \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W_{xh}} 
$$

Further, we can take the derivative with respect to $W_{xh}$ over the whole sequence as :

$$
\frac{\partial L} {\partial W_{xh}} = \sum_{t}^{T} \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W_{xh}} 
$$

Do not forget that $\frac{\partial h_{t+1}}{\partial h_{k}}$ is a chain rule in itself, again!

# REFERENCES
1. [http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/){:target="_blank"}
2. [https://arxiv.org/abs/1610.02583](https://arxiv.org/abs/1610.02583){:target="_blank"}
3. [https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf){:target="_blank"}