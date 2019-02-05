---
layout: post
title: "Recurrent Neural Networks and gradient related problems"
author: "MMA"
comments: true
---

Up to now we have mostly looked at feedforward neural networks, where the activations flow only in one direction, from the input layer to the output layer. A recurrent neural network (RNN) looks very much like a feedforward neural network, except it also has connections pointing backward.

RNN, composed of just one neuron receiving inputs, producing an output, and sending that output back to itself (left). At each time step $t$ (also called a frame), this recurrent neuron receives the inputs $x_{t}$ as well as its own output from the previous time step, $y_{t–1}$. We can represent this tiny network against the time axis (right). This is called *unrolling the network through time*.

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/rnn.png)

Here, each layer on the RNN represents a distinct time step and the weights are shared across time and this property helps reducing the number of parameters!

An illustration of the RNN model is given below:
![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/rnn_full_model.png)

The dynamical system is defined by:

$$
\begin{split}
    h_{t} & = f_{h} (X_{t}, h_{t-1})\\
    y_{t} &= f_{o}(h_{t})
\end{split}
$$

A conventional RNN is constructed by defining the transition function and the output function for a single instance:

$$
\begin{split}
    h_{t} & = f_{h} (X_{t}, h_{t-1}) = \phi_{h}(W_{xh}^{T} \cdot X_{t} + W_{hh}^{T}\cdot h_{t-1} +b_{h})\\
    y_{t} &= f_{o}(h_{t}) = \phi_{o}(W_{yh}^{T}\cdot h_{t} + b_{y})
\end{split}
$$

where $W_{xh}$, $W_{hh}$ and $W_{yh}$ are respectively the input, the transition (hidden state), and the output weight matrices and $\phi_{h}$ and $\phi_{o}$ are element-wise nonlinear functions. It is usual to use a saturating nonlinear function such as sigmoid function or a hyperbolic tangent function for $\phi_{h}$. $\phi_{o}$  is generally softmax activation for classification problem. 

Just like for feedforward neural networks, we can compute a recurrent layer’s output in one shot for a whole mini-batch by placing all the inputs at time step $t$ in an input matrix $X_{t}$:

$$
\begin{split}
    h_{t} & = \phi_{h}(X_{t}\cdot W_{xh} + h_{t-1}\cdot  W_{hh} + b_{h})\\
    &= \phi_{h}( [X_{t} h_{t-1}] \cdot W + b_{h})\\
    y_{t} &= \phi_{o}(h_{t}\cdot W_{yh} + b_{y})
\end{split}
$$

From the above equations we can see that the RNN model is parameterized by three weight matrices.

Let's denote $m$ as the number of instances in the mini-batch, $n_{neurons}$ as the number of neurons, and $n_{inputs}$ as the number of input features.

1. $X_{t}$ is an $m \times n_{inputs}$ matrix containing the inputs for all instances.
2. $h_{t-1}$ is an $m \times n_{neurons}$ matrix containing the hidden state of the previous time-step for all instances.
3. $W_{xh}$ is an $n_{inputs} \times n_{neurons}$ matrix containing the connection weights between input and the hidden layer.
4. $W_{hh}$ is an $n_{neurons} \times n_{neurons}$ matrix containing the connection weights between two hidden layers.
5. $W_{yh}$ is an $n_{neurons} \times n_{neurons}$ matrix containing the connection weights between the hidden layer and the output.
6. $b_{h}$ is a vector of size $n_{neurons}$ containing each neuron’s bias term.
7. $b_{y}$ is a vector of size $n_{neurons}$ containing each output’s bias term.
8. $y_{t}$ is an $m \times n_{neurons}$ matrix containing the layer’s outputs at time step $t$ for each instance in the mini-batch 
9. The weight matrices $W_{xh}$ and $W_{yh}$ are often concatenated vertically into a single weight matrix $W$ of shape $(n_{inputs} +  n_{neurons}) \times  n_{neurons}$.
10. The notation $[X_{t} h_{t-1}]$ represents the horizontal concatenation of the matrices $X_{t}$ and $h_{t-1}$, shape of $m \times (n_{inputs} +  n_{neurons})$

In literature, $\phi_{h}$ is chosen to be the hyperbolic tangent function which is is the non-linearity added to the hidden states while $\phi_{o}$ is softmax activation function used in the output layer.

RNNs are trained in a sequential supervised manner. For time step $t$, the error is given by the difference between the predicted and targeted: $(\hat{y}_{t}−y_{t})$. The overall loss $L(\hat{y}_{t}−y_{t})$ is usually a sum of time step specific losses found in the range of intrest $[t,T]$ given by:

$$
L (\hat{y}, y) = \sum_{t = 1}^{T} L(\hat{y}_{t}, y_{t}) 
$$

## Backpropagation Through Time
![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/BPTT.png)

Training of the unfolded recurrent neural network is done across multiple time steps using backpropagation where the overall error gradient is equal to the sum of the individual error gradients at each time step. The red lines in the image is where we calculate the gradients.

This algorithm is known as backpropagation through time (BPTT). If we take a total of $T$ time steps, the error is given by the following equation:

$$
\frac{\partial \textbf{E}}{\partial \textbf{W}} = \sum_{t=1}^{T} \frac{\partial \textbf{E}_{t}}{\partial \textbf{W}}
$$

Applying chain rule to compute the overall error gradient we have the following

$$4
\frac{\partial \textbf{E}}{\partial \textbf{W}} = \sum_{t=1}^{T} \frac{\partial \textbf{E}}{\partial \textbf{y}_{t}} \frac{\partial \textbf{y}_{t}}{\partial \textbf{h}_{t}} \overbrace{\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}}}^{ \bigstar } \frac{\partial \textbf{h}_{k}}{\partial \textbf{W}} $$

The term marked $\bigstar$, i.e., $\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}}$, is the derivative of the hidden state at time $t$ with respect to the hidden state at time $k$.  This term involves products of Jacobians $\frac{\partial \textbf{h}_{i}}{\partial \textbf{h}_{i-1}}$ over subsequences linking an event at time $t$ and one at time $k$ given by:

$$
\begin{split}
\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}} &= \frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{t-1}} \frac{\partial \textbf{h}_{t-1}}{\partial \textbf{h}_{t-2}} \cdots \frac{\partial \textbf{h}_{k+1}}{\partial \textbf{h}_{k}}  \\
&= \prod_{i=k+1}^{t} \frac{\partial \textbf{h}_{i}}{\partial \textbf{h}_{i-1}}
\end{split}
$$

The product of Jacobians in Equation above features the derivative of the term $h_{t}$ with respect to $h_{t-1}$ (i.e., \frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{t-1}}) which when evaluated on RNN definition ($h_{t} = f_{h} (X_{t}, h_{t-1}) = \phi_{h}(W_{xh}^{T} \cdot X_{t} + W_{hh}^{T}\cdot h_{t-1} +b_{h})$) yields $W_{hh}^{T}\left[f^{'}(h_{t-1}) \right]$, therefore:

# Vanishing and Exploding Gradients
The backpropagation algorithm works by going from the output layer to the input layer, propagatinf the error gradient on the way back. Once the algorithm has computed the gradient of the cost function with regards to each parameter in the network, it uses these gradients to update each parameter with a Gradient Descent step. Unfortunately, sometimes, gradients get smaller and smaller as the algorithm progresses back to the lower layers. As a result, gradient descent update leaves the lower connection weights unchanged (relatively) and training never converges to a good solution. However, lower layers (sometimes called earlier layers) in the network are important because they are responsible to learn and detecting the simple patterns and are actually the building blocks of the network. Obviously, if they give improper and inaccurate results, then the next layers and thus, the complete network, will not perform nicely and produce accurate results. This phenomenon is called *vanishing gradient problem*. In some cases, the opposire can happen: the gradients can grow bigger and bigger, so many layers get insanely large weight updates and the algorithm diverges, This is *exploding gradients* problem. Both problems are mostly encountered in recurrent nural networks. There are two factors that affect the magnitude of gradients - the weights and the activation functions (or more precisely, their derivatives) that the gradient passes through. Therefore, choosing a proper weight initialization and activation function play an important role for the model we are training on. 

# Activation Functions

## Sigmoid Function

## Hyperbolic Tangent Function

## Rectified Linear Unit (ReLU) 
ReLU is a piecewise non-linear function that corresponds to:

$$
ReLU(x) = \begin{cases} 0 & \mbox{if $x < 0$}\\ x & \mbox{if $x \geq 0$} \end{cases}
$$

Another way of writing the ReLU function is like so:

$$
ReLU(x) = max(0, x)
$$

where  $x$ is the input to a neuron. In other words, when the input is smaller than zero, the function will output zero. Else, the function will mimic the identity function. So, the range of ReLU is between $0$ to $\infty$.

The derivative of ReLU function is given by:

$$
\frac{d}{d x} ReLU(x) = \begin{cases} 0 & \mbox{if $x < 0$}\\ 1 & \mbox{if $x \geq 0$} \end{cases}
$$

{% highlight python %}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

def relu(z):
    return np.maximum(0, z)

def derivative_relu(z):
    return (z >= 0).astype(z.dtype)

z = np.linspace(-5, 5, 200)

plt.figure()
plt.plot(z, derivative_relu(z), "g-.", linewidth=2, label="ReLU Derivation")
plt.grid(True)
plt.legend(loc='upper left', fontsize=12)
plt.title("Derivative of ReLU Activation Function", fontsize=14)
plt.savefig('relu_derivative.png')

plt.figure()
plt.plot(z, relu(z), "b-.", linewidth=2, label="ReLU Derivation")
plt.grid(True)
plt.legend(loc='upper left', fontsize=12)
plt.title("ReLU Activation Function", fontsize=14)
plt.savefig('relu.png')
{% endhighlight %}

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/relu.png)

ReLU is cheap to compute because it has an easy function and a simple derivative. It converges faster. It does not saturate. It doesn’t have the vanishing gradient problem suffered by other activation functions like sigmoid or tanh. It introduces sparsity in the network. Sparsity results in concise models that often have better predictive power and less overfitting/noise. A sparse network is faster than a dense network, as there are fewer things to compute.

Unfortunately, ReLU also suffers several drawbacks, for instance, ReLU units can be fragile during training and can “die”. As can be seen easily in the definition, ReLU neurons output zero and have zero derivatives for all negative inputs. So, if the weights in the network always lead to negative inputs into a ReLU neuron, that neuron is effectively not contributing to the network's training. Mathematically, the gradient contribution to the weight updates coming from that neuron is always zero. A "dead" ReLU always outputs the same value for any input. Probably this is arrived at by learning a large negative bias term for its weights. Once a ReLU ends up in this state, the neuron is unlikely to come back to life. A fix is to modify the flat side to have a small gradient.

Implementing ReLU in TensorFlow is trivial, just specify the activation function when building each layer using `tf.nn.relu`.

### ReLU Variants

#### Leaky ReLU & Parametric ReLU (PReLU)

Leaky ReLU has a small slope for negative values, instead of altogether zero.
$$
LeakyReLU(x) = max(\alpha x, x)
$$

The hyperparameter $\alpha$ defines how much the function "leaks": it is the slope of the function for $x <0$ and is typically set to $0.01$. This small slope ensures that leaky ReLUs never die; they can go into a long coma but they have a chance to eventually wake up.

{% highlight python %}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline


def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

z = np.linspace(-5, 5, 200)
plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [-0.5, 4.2], 'k-')
plt.grid(True)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])

plt.savefig("leaky_relu_plot")
plt.show()
{% endhighlight %}

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/leaky_relu_plot.png)

Parametric ReLU (PReLU) is a type of leaky ReLU that, instead of having $\alpha$ predetermined slope like $0.01$, $\alpha$ is to be learned during training. It is reported that PReLU strongly outperform ReLU on large image datasets but on smaller datasets, it runs the risk of overfitting the training set.

Implementing Leaky ReLU in TensorFlow is trivial, just specify the activation function when building each layer using `tf.nn.leaky_relu`.

Similarly, implementing Parametric ReLU in TensorFlow is trivial, just specify the activation function when building each layer using `tf.keras.layers.PReLU`.

#### Exponential Linear (ELU, SELU)
Similar to leaky ReLU, ELU has a small slope for negative values. Instead of a straight line, it uses a log curve. ELU outperformed all the ReLU variants in the original paper's experiments: training time was reduced and the neural network performed better on the test set. ELU is given by:

$$
ELU(x) =\begin{cases} \alpha (exp(x)-1) & \mbox{if $x < 0$}\\ x & \mbox{if $x \geq 0$} \end{cases}
$$

{% highlight python %}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

z = np.linspace(-5, 5, 200)
plt.plot(z, elu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

plt.savefig("elu_plot")
plt.show()
{% endhighlight %}

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/elu_plot.png)

The main drawback of the ELU activation function is that it is slower to compute than the ReLU and its variants due to the use of the exponential function but during training this is compensated by the faster convergence rate. However, at test time, an ELU network will be slower than a ReLU network. 

Implementing ELU in TensorFlow is trivial, just specify the activation function when building each layer using `tf.nn.elu`.

Scaled exponential linear unit (SELU) has been also proposed in the literature. SELU is some kind of ELU but with a little twist. $\alpha$ and $\lambda$ are two pre-defined constants, meaning we don’t backpropagate through them and they are not hyperparameters to make decisions about. $\alpha$ and $\lambda$ are derived from the inputs. $\lambda$ is called as a scale parameter. Essentially, SELU is `scale * elu(x, alpha)`. 

$$
SELU(x) = \lambda \begin{cases} \alpha (exp(x)-1) & \mbox{if $x < 0$}\\ x & \mbox{if $x \geq 0$} \end{cases}
$$

SELU can’t make it work alone. Thus, a custom weight initialization technique is being used. It is to be used together with the initialization "lecun_normal", which it draws samples from a truncated normal distribution centered on $0$ with `stddev <- sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight tensor.

For standard scaled inputs (mean 0, stddev 1), the values are $\alpha \approx 1.6732$, $\lambda \approx 1.0507$. Let's plot and see what it looks like for these values.

{% highlight python %}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

def selu(z, alpha=1.6732632423543772848170429916717, lamb=1.0507009873554804934193349852946):
    return np.where(z < 0, lamb * alpha * (np.exp(z) - 1), z)

z = np.linspace(-5, 5, 200)
plt.plot(z, selu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"SELU activation function ($\alpha \approx 1.6732$ and $\lambda \approx 1.0507$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

plt.savefig("selu_plot")
plt.show()
{% endhighlight %}

![Placeholder Image](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/selu_plot.png)

Implementing SELU in TensorFlow is trivial, just specify the activation function when building each layer using `tf.nn.selu`.



# Proposed Solutions For Exploding Gradients

# Proposed Solutions For Vanishing Gradients

# REFERENCES
1. [Paul J. Werbos. Backpropagation through time: what it does and how to do it.Proceedings of the IEEE, 78(10): 1550 – 1560, 1990.](http://axon.cs.byu.edu/~martinez/classes/678/Papers/Werbos_BPTT.pdf){:target="_blank"}

2. [https://www.utc.fr/~bordesan/dokuwiki/_media/en/glorot10nipsworkshop.pdf](https://www.utc.fr/~bordesan/dokuwiki/_media/en/glorot10nipsworkshop.pdf){:target="_blank"}

3. [https://arxiv.org/abs/1511.07289](https://arxiv.org/abs/1511.07289){:target="_blank"}

4. [https://arxiv.org/abs/1706.02515](https://arxiv.org/abs/1706.02515){:target="_blank"}