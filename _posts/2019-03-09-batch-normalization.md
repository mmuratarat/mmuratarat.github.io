---
layout: post
title: "Batch Normalization"
author: "MMA"
comments: true
---

Normalizing the input of your network is a well-established technique for improving the convergence properties of a network. A few years ago, a technique known as [batch normalization](https://arxiv.org/abs/1502.03167){:target="_blank"} was proposed to extend this improved loss function topology to more of the parameters of the network.

The method consists of adding an operation in the model just before the activation function of each layer, simply, zero-centering and normalizing the inputs of that patricular layer and then scaling and shifting the result using two new parameters per layer (one for scaling, the other for shifting). In other words, this operation lets the model learn the optimal scale and mean of the inputs fir each layer.

In order to zero-center and normalize the inputs, the algorithm needs to estimate the inputs' mean and standard deviation. It does so by evaluating the mean and standard deviation of the inputs over the current mini-bathc (hence the name "Batch Normlization"). 

It is a widely used technique for normalizing the internal representation of data on models that can lead to substantial reduction in convergence time.

It introduces some noise into the network, so it can regularize the model a bit. However, regularization is a side effect of BN, rather than the main objective. 

However, there is a one drawback of batch normalization, which is that it makes the training slower due to extra computations required at each layer.

## Batch Normalization Equations

A batch normalization layer is given a batch of $N$ examples, each of which is a $D$-dimensional vector in a mini-batch $\phi$, where $D$ is the number of hidden units. We can represent the inputs as a matrix $X \in R^{N \times D}$ where each row $x_{i}$ is a single example. Each example $x_{i}$ is normalized by
 
\begin{equation}
    \begin{split}
        {\mu_\phi} &\leftarrow {\frac{1}{N}}{\sum_{i=1}^N}x_i \qquad \mathsf(mini-batch\ mean)\\
        {\sigma^2_\phi} &\leftarrow {\frac{1}{N}}{\sum_{i=1}^N} {(x_i - {\mu_\phi})^2} \qquad \mathbf(mini-batch\ variance)\\
        {\hat{x_i}} &\leftarrow \frac{x_i-{\mu_\phi}}{\sqrt{\sigma^2_\phi + \epsilon}} \qquad \mathbf(\text{An affine transform - normalize})
    \end{split}
\end{equation}

Every component of $\hat{x}$ has zero mean and unit variance. However, we want hidden units to have different distributions. In fact, this would perform poorly for some activation functions such as the sigmoid function. Thus, we'll allow our normalization scheme to learn the optimal distribution by scaling our normalized values by $\gamma$ and shifting by $\beta$:

\begin{equation}
    {y_i}\leftarrow {\gamma \cdot \hat{x_i}} + \beta \equiv BN_{\gamma,\beta}{(x_i)}\qquad \mathbf(scale \ and \ shift)
\end{equation}

In other words, we've now allowed the network to normalize a layer into whichever distribution is most optimal for learning.

* $\mu_\phi \in R^{1 \times D}$: is the empirical mean of each input dimension across the whole mini-batch. 
* $\sigma_\phi \in R^{1 \times D}$ is the empirical standard deviation of each input dimension across the whole mini-batch.
* $N$ is the number of instances in the mini-batch
* $\hat{x_i}$ is the zero-centered and normalized input.
* $\gamma \in R^{1 \times D}$ is the scaling parameter for the layer.
* $\beta \in R^{1 \times D}$ is the shifting parameter (offset) for the layer.
* $\epsilon$ is added for numerical stability, just in case ${\sigma_\phi}^2$ turns out to be 0 for some estimates. This is also called a smoothing term.
* $y_i$ is the output of the BN operation. It is the scaled and shifted version of the inputs. ${y_i}=BN_{\gamma,\beta}{(x_i)}}$

For notational simplicity, we can express the entire layer as

\begin{equation}
\begin{split}
  \hat{X} &= \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
  Y &= \gamma \odot \hat{X} + \beta
\end{split}
\end{equation}

where $\odot$ denotes the Hadamard (element-wise) product. In the case of $\gamma \odot \hat{X}$, where $Y$ is a row vector and $\hat{X}$ is a matrix, each row of $\hat{X}$ is multiplied element-wise by $\gamma$.

**NOTE**:  the authors did something really clever. They realized that by normalizing the output of a layer they could limit its representational power and, therefore, they wanted to make sure that the Batch Norm layer could fall back to the identity function. If $\beta$ is set to $\mu_\phi$ and $\gamma$ to $\sqrt{\sigma^2_\phi + \epsilon}$, $\hat{x_i}$ equals to $x_i$,  thus working as an identity function. That means that the network has the ability to ignore the Batch Norm layer if that is the optimal thing to do and introducing batch normalization alone would not reduce the accuracy because the optimizer still has the option to select no normalization effect using the identity function and it would be used by the optimizer to only improve the results.

**NOTE**: $\gamma$ and $\beta$ are learnable parameters that are initialized with $\gamma =1$ and $\beta = 1$.

**NOTE**: Batch Normalization is done individually at every hidden unit. 

### Inference
During testing time, the implementation of batch normalization is quite different.

While it is acceptable to compute the mean and variance on a mini-batch when we are training the network, the same does not hold on test time. When the batch is large enough, its mean and variance will be close to the population’s. The non-deterministic mean and variance has regularizing effects.On test time, we want the model to be deterministic. 

At test time, there is no mini-batch to compute the empirical mean and standard deviation, so instead you simply use the whole training set's mean and standard deviation, meaning that population instead of the mini-batch statistics.

we calculate “population average” of mean and variances after training, using all the batches' means and variances, and at inference time, we fix the mean and variance to be this value and use it in normalization. This provides more accurate value of mean and variance.

\begin{equation}
    \begin{split}
        E(x) &\leftarrow E_{\phi}(\mu_\phi)\\
        Var(x)&\leftarrow \frac{m}{m-1} E_{\phi}(\sigma^2_\phi) \quad (\text{unbiased variance estimate})
    \end{split}
\end{equation}

Then, at inference time, using those population mean and variance, we do normalization:
\begin{equation}
    \begin{split}
        \hat{x} &= \frac{x - E(x)}{\sqrt{Var(x) + \epsilon}}\\
        y &= \gamma \hat{x} + \beta \\
        y &= \gamma \frac{x - E(x)}{\sqrt{Var(x) + \epsilon}} + \beta\\
        y &= \frac{\gamma x}{\sqrt{Var(x) + \epsilon}} + \left(\beta - \frac{\gamma E(x)}{\sqrt{Var(x) + \epsilon}} \right)
    \end{split}
\end{equation}

But, sometimes, it is difficult to keep track of all the mini-batch mean and variances. In such cases, exponentially weighted "moving average" can be used as global statistics to update population mean and variance:

\begin{equation}
    \begin{split}
    \mu_{mov} &= \alpha \mu_{mov} + (1-\alpha) \mu_\phi\\
    \sigma^2_{mov} &= \alpha \sigma^2_{mov} + (1-\alpha) \sigma^2_\phi
    \end{split}
\end{equation}

Here $\alpha$ is the "momentum" given to previous moving statistic, around $0.99$, and those with $\phi$ subscript are mini-batch mean and mini-batch variance. This is the implementation found in most libraries, where the momentum can be set manually.

## Backward Pass
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/batch-norm-computational-graph.png?raw=true)
We follow the dark arrows for the forward pass and then we backpropagate the error using the red ones. The dimension of the output of each node is displayed on top of each node.

Let $L$ be the loss function and we are given $\frac{\partial L}{\partial Y} \in R^{N \times D}$, the gradient of loss with respect to $Y$. Our goal is to compute three gradients:

1. $\frac{\partial L}{\partial \gamma} \in R^{1 \times D}$ to perform gradient descent update on $\gamma$.
2. $\frac{\partial L}{\partial \beta} \in R^{1 \times D}$ to perform gradient descent update on $\beta$.
3. $\frac{\partial L}{\partial X} \in R^{N \times D}$ to pass on the gradient signal to lower layers.

Both $\frac{\partial L}{\partial \gamma}$ and $\frac{\partial L}{\partial \beta}$ are straightforward to compute. Let $y_{i}$ be the $i$-th row of $Y$.

\begin{equation}
\begin{split}
\frac{\partial L}{\partial \gamma} &= \frac{\partial L}{\partial y_{i}} \frac{\partial y_{i}}{\partial \gamma}\\
&=\sum_{i=1}^{N} \frac{\partial L}{\partial y_{i}} \cdot \hat{x}_{i}
\end{split}
\end{equation}

Notice that we sum from 1 to N, because we are working with batches.

Similarly, we compute the gradient with respect to $\beta$ as follows:

\begin{equation}
\begin{split}
\frac{\partial L}{\partial \beta} &= \frac{\partial L}{\partial y_{i}} \frac{\partial y_{i}}{\partial \beta}\\
&=\sum_{i=1}^{N} \frac{\partial L}{\partial y_{i}} 
\end{split}
\end{equation}

What we need to compute next is the partial derivative of the loss with respect to the input $x_{i}$. So, the previous layers can compute their gradients and update their parameters. We need to gather all the expressions where $x_{i}$ is used that has influence on $y_{i}$. 

$x_{i}$ is used to compute $\hat{x}_{i}$, $\mu_\phi$ and $\sigma_\phi$. Therefore,

\begin{equation}
\begin{split}
\frac{\partial L}{\partial x_{i}} &= \frac{\partial L}{\partial \hat{x}_{i}} \frac{\partial \hat{x}_{i}}{\partial x_{i}} \\
& +  \frac{\partial L}{\partial \sigma^{2}_\phi}\frac{\partial \sigma^{2}_\phi}{\partial x_{i}}\\
&+ \frac{\partial L}{\partial \mu_\phi}\frac{\partial \mu_\phi}{\partial x_{i}}
\end{split}
\end{equation}

Let's compute and simplify each of these terms individually

The first one is the easiest to derive:

\begin{equation}
\frac{\partial L}{\partial \hat{x}_{i}} =  \frac{\partial L}{\partial y_{i}}  \cdot \gamma
\end{equation}

and 

\begin{equation}
\frac{\partial \hat{x}_{i}}{\partial x_{i}} = \left(\sigma^2_\phi + \epsilon \right)^{-1/2}
\end{equation}

Therefore,

\begin{equation}
\frac{\partial L}{\partial \hat{x}_{i}} \frac{\partial \hat{x}_{i}}{\partial x_{i}} =  \frac{\partial L}{\partial y_{i}}  \cdot \gamma \cdot \left(\sigma^2_\phi + \epsilon \right)^{-1/2}
\end{equation}

The very next expression is a bit longer.

\begin{equation}
\frac{\partial L}{\partial \sigma^2_\phi} = \frac{\partial L}{\partial \hat{x}_{i}} \frac{\partial \hat{x}_{i}}{\partial \sigma^2_\phi}
\end{equation}

We know that $\hat{x_i} = \frac{x_i-{\mu_\phi}}{\sqrt{\sigma^2_\phi + \epsilon}}$. Here $(x_i-{\mu_\phi})$ is constant, so:

\begin{equation}
\begin{split}
\frac{\partial L}{\partial \sigma^2_\phi} &= \sum_{i=1}^{N}\frac{\partial L}{\partial \hat{x}_{i}} \frac{\partial \hat{x}_{i}}{\partial \sigma^2_\phi}\\
&= \sum_{i=1}^{N} \frac{\partial L}{\partial y_{i}} \cdot \gamma \cdot (x_i-{\mu_\phi})\left(\frac{-1}{2} \right) \left(\sigma^2_\phi + \epsilon \right)^{-3/2}\\
&=-\frac{\gamma \left(\sigma^2_\phi + \epsilon \right)^{-3/2}}{2} \sum_{i=1}^{N} \frac{\partial L}{\partial y_{i}} (x_i-{\mu_\phi}) 
\end{split}
\end{equation}

As what happened with the gradients of $\gamma$ and $\beta$, to compute the gradient of $\sigma^2_\phi$, we need to sum over the contributions of all elements from the batch. The same happens to the gradient of $\mu_\phi$ as it is also a $D$-dimensional vector. However, this time, $\sigma^2_\phi$ is also a function of $\mu_\phi$.

\begin{equation}
\begin{split}
\frac{\partial L}{\partial \mu_\phi} &= \frac{\partial L}{\partial \hat{x}_{i}}\frac{\partial \hat{x}_{i}}{\partial \mu_\phi}\\
&+ \frac{\partial L}{\partial \sigma^2_\phi}\frac{\partial \sigma^2_\phi}{\partial \mu_\phi}
\end{split}
\end{equation}

Let's compute the missing partials one at a time.

From

\begin{equation}
\hat{x_i} = \frac{x_i-{\mu_\phi}}{\sqrt{\sigma^2_\phi + \epsilon}}
\end{equation}

we compute

\begin{equation}
\frac{\partial \hat{x}_{i}}{\partial \mu_\phi} = \frac{1}{\sqrt{\sigma^2_\phi + \epsilon}} \cdot (-1)
\end{equation}

and from

\begin{equation}
\sigma^2_\phi = {\frac{1}{N}}{\sum_{i=1}^N} {(x_i - {\mu_\phi})^2}
\end{equation}

we calculate

\begin{equation}
\frac{\partial \sigma^2_\phi}{\partial \mu_\phi}= \frac{-2}{N} \sum_{i=1}^{N} (x_i-{\mu_\phi})
\end{equation}

We already know $\frac{\partial L}{\partial \hat{x}_{i}}$ and $\frac{\partial L}{\partial \sigma^2_\phi}$, so, let's put them all together:

\begin{equation}
\begin{split}
\frac{\partial L}{\partial \mu_\phi} &= \sum_{i=1}^N \frac{\partial L}{\partial y_{i}}  \cdot \gamma \cdot \frac{1}{\sqrt{\sigma^2_\phi + \epsilon}} \cdot (-1)\\
&+ \left(-\frac{\gamma \left(\sigma^2_\phi + \epsilon \right)^{-3/2}}{2} \sum_{i=1}^{N} \frac{\partial L}{\partial y_{i}} (x_i-{\mu_\phi})  \cdot \frac{-2}{N} \sum_{i=1}^{N} (x_i-{\mu_\phi}) \right)
\end{split}
\end{equation}

It seems complicated but it is actually super easy to simplify. Since $\sum_{i=1}^{N} (x_{i} - \mu_\phi) = 0$, the second term of this expression will be 0. Then,

\begin{equation}
\begin{split}
\frac{\partial L}{\partial \mu_\phi} &= -\gamma \cdot \left(\sigma^2_\phi + \epsilon \right)^{-1/2} \sum_{i=1}^N \frac{\partial L}{\partial y_{i}}   
\end{split}
\end{equation}

Now, we can easily compute:

\begin{equation}
\begin{split}
\frac{\partial L}{\partial x_{i}} &= \frac{\partial L}{\partial \hat{x}_{i}} \frac{\partial \hat{x}_{i}}{\partial x_{i}} \\
& +  \frac{\partial L}{\partial \sigma^{2}_\phi}\frac{\partial \sigma^{2}_\phi}{\partial x_{i}}\\
&+ \frac{\partial L}{\partial \mu_\phi}\frac{\partial \mu_\phi}{\partial x_{i}}
\end{split}
\end{equation}

We still have some missing gradients which are really easy to calculate:


\begin{equation}
\frac{\partial \sigma^{2}_\phi}{\partial x_{i}} = \frac{2(x_i - {\mu_\phi})}{N}
\end{equation}

since 

\begin{equation}
{\sigma^2_\phi} = {\frac{1}{N}}{\sum_{i=1}^N} {(x_i - {\mu_\phi})^2}
\end{equation}

and

\begin{equation}
\frac{\partial \mu_\phi}{\partial x_{i}} = \frac{1}{N}
\end{equation}

since

\begin{equation}
{\mu_\phi} = {\frac{1}{N}}{\sum_{i=1}^N}x_i
\end{equation}

So,

\begin{equation}
\frac{\partial L}{\partial x_{i}} = \frac{\partial L}{\partial \hat{x}_{i}}  \frac{1}{\sqrt{\sigma^2_\phi + \epsilon}} +  \frac{\partial L}{\partial \sigma^{2}_\phi} \frac{2(x_i - {\mu_\phi})}{N} + \frac{\partial L}{\partial \mu_\phi}\frac{1}{N}
\end{equation}

## Batch Normalization in Tensorflow

When training, the moving mean and moving variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be executed alongside the train_op. Also, be sure to add any batch normalization ops before getting the update_ops collection. Otherwise, update_ops will be empty, and training/inference will not work properly. For example:

{% highlight python %} 
import tensorflow as tf
# ...
# declare a placeholder to tell the network if it is in training time or inference time 
is_train = tf.placeholder(tf.bool, name="is_train");
# ...
x_norm = tf.layers.batch_normalization(x, 
                                       center=True,
                                       scale=True,
                                       training=is_train)
# ...
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
{% endhighlight %}

**NOTE**: In Tensorflow, moving mean is initialized from zero and moving variance is initialized from one. 

## Where to Insert Batch Norm Layers?
Batch normalization may be used on the inputs to the layer before or after the activation function in the previous layer. It may be more appropriate after the activation function if for s-shaped functions like the hyperbolic tangent and logistic function. It may be appropriate before the activation function for activations that may result in non-Gaussian distributions like the rectified linear activation function, the modern default for most network types, as the authors of the original paper puts: 'The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution'.

## Advantages of Batch Normalization
* reduces internal covariate shift
* improves the gradient flow
* more tolerant to saturating nonlinearities because it is all about the range of values fed into those activation functions. 
* reduces the dependence of gradients on the scale of the parameters or their initialization (less sensitive to weight initialization)
* allows higher learning rates because of less danger of exploding/vanishing gradients.
* acts as a form of regularization
* all in all accelerates neural network convergence.

## Don’t Use With Dropout
Batch normalization offers some regularization effect, reducing generalization error, perhaps no longer requiring the use of dropout for regularization. Further, it may not be a good idea to use batch normalization and dropout in the same network. The reason is that the statistics used to normalize the activations of the prior layer may become noisy given the random dropping out of nodes during the dropout procedure.

# REFERENCES
1. [https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82](https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82){:target="_blank"}
2. [https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html){:target="_blank"}