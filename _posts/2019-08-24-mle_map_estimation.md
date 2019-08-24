---
layout: post
title: "Maximum Likelihood Estimation and Maximum A Posteriori Estimation"
author: "MMA"
comments: true
---

# Maximum Likelihood Estimation (MLE)

Consider $n$ i.i.d. random variables $x_{1}, x_{2}, ..., x_{n}$.

$x_{i}$ is a sample from density estimation $P(x_{i} \mid \theta)$. Then, we can write the likelihood function as follows:

\begin{equation}
L(\theta) = \prod_{i} $P(x_{i} \mid \theta)
\end{equation}

If we take the logarithm of it and find the maximum likelihood estimator of $\theta$:

\begin{equation}
\begin{split}
LL(\theta) = \log L(\theta) &= \log \left(\prod_{i} P(x_{i} \mid \theta) \right)\\
&= \sum_{i=1} log\left(P(x_{i} \mid \theta) \right)
\end{split}
\end{equation}

\begin{equation}
\theta_{MLE} = \underset{\theta}{\operatorname{argmax}} LL(\theta) = \underset{\theta}{\operatorname{argmax}} \sum_{i=1} log\left(P(x_{i} \mid \theta) \right)
\end{equation}

Here, $\theta_{MLE}$ is the exact value of $\theta$ which maximizes the log-likelihood. The value of $\theta$ that maximizes the log-likelihood can be obtained by having the derivative of the log-likelihood function with respect to $\theta$ and setting it to 0. 

# Bayes Theorem

Bayes Theorem states that

\begin{equation}
P(\theta \mid X) = \frac{P(X \mid \theta) \dot P(\theta)}{P(X)}
\end{equation}

where $P(X \mid \theta)$ is the probability distribution of of the data given the parameter $\theta$ (sometimes called the likelihood) and $P(X)$ is the prior distribution which represents our beliefs about the uncertain parameter $\theta$ before seeing any data. It is the distribution over the parameter $\theta$. $P(\theta \mid X)$ is the posterior distribution which is the belief about the model after seeing the data. After $X$ is observed, we have the posterior distribution over the parameters $\theta$ conditional on data. We use this distribution to predict the new data. Here $P(X)$ is the normalizing constant, does not depend on $\theta$, which forces $P(\theta \mid X)$ to integrate to 1, so we can safely ignore it. 

# Maximum A Posteriori (MAP) Estimator

Recall that maximum likelihod estimate of $\theta$ is given by:

\begin{equation}
\theta_{MLE} = \underset{\theta}{\operatorname{argmax}} \sum_{i=1} log\left(P(x_{i} \mid \theta) \right)
\end{equation}

Then, similarly, MAP estimator of $\theta$  is,

\begin{equation}
\theta_{MAP} = \underset{\theta}{\operatorname{argmax}} P(\theta \mid X) = \underset{\theta}{\operatorname{argmax}} P(\theta \mid x_{1}, x_{2},...,x_{n}) = \underset{\theta}{\operatorname{argmax}} \frac{P(x_{1}, x_{2},...,x_{n} \mid \theta) \dot P(\theta)}{P(x_{1}, x_{2},...,x_{n})}
\end{equation}

Since $x_{1}, x_{2},...,x_{n}$ is i.i.d. random variables,

\begin{equation}
\begin{split}
\theta_{MAP} &= \underset{\theta}{\operatorname{argmax}}\frac{P(x_{1} \mid \theta)P(x_{2} \mid \theta)...P(x_{n} \mid \theta)P(\theta)}{P(x_{1})P(x_{2})...P(x_{n})}\\
&=\underset{\theta}{\operatorname{argmax}} \frac{\prod_{i=1} P(x_{i} \mid \theta)P(\theta)}{\prod_{i=1}P(x_{i})}
\end{split}
\end{equation}

where $P(\theta)$ is the prior distribution of $\theta$, We can ignore the denominator.

As before, can often be more convenient to use logarithm,

\begin{equation}
\theta_{MAP} = \underset{\theta}{\operatorname{argmax}} \sum_{i=1} log \left(P(x_{i} \mid \theta)\right) + log \left( P(\theta)\right)
\end{equation}

As one can see easily, MAP estimate is a special case of MLE, which uses the mode as a point estimate of posterior distribution.

For computational simplicity, specific probability distributions are used corresponding to the probability distribution of the likelihood. It is called conjugate prior distribution. 

# An Example
Let's say that a random variable $X$ follows binomial distribution with parameter $\theta$. Let $x_{1}, x_{2},...,x_{n}$ be the independent random samples of $X$. Recall that the probability density function for the Binomial distribution with parameter $\theta$

\begin{equation}
P(X \mid \theta) = {n\choose X} \theta^{X} (1-\theta)^{n-X},\,\,\,\, X=0,1,2,...n
\end{equation}

Let's write down the likelihood function,

\begin{equation}
P(\theta \mid X) = L(\theta) = \prod_{i=1} P(x_{i} \mid \theta) = \prod_{i=1} {{n}\choose{x_{i}}} \theta^{x_{i}} (1-\theta)^{n-x_{i}}
\end{equation}

Taking the natural logarithm on both sides gives:
\begin{equation}
log L(\theta) =  \sum_{i=1} log \left(P(x_{i} \mid \theta)\right) = \sum_{i=1} \left[log {{n}\choose{x_{i}}} + x_{i} log (\theta) + (n-x_{i}) log(1-\theta)\right]
\end{equation}

Since $log L(\theta)$ is a continuous function of $\theta$, it has a maximum value. This value can be found by
taking the derivative of $log L(\theta)$ with respect to $\theta$, and setting it equal to 0. So,

\begin{equation}
\theta_{MLE} = \frac{\partial log L(\theta)}{\partial \theta} = \frac{1}{\theta} \sum_{i=1}x_{i} - \frac{1}{1-\theta} \left(n - \sum_{i=1}x_{i}  \right) = 0
\end{equation}

gives that $\theta_{MLE} = \frac{\sum_{i=1}x_{i}}{\theta} = \frac{X}{\theta}$ where $X$ is the total number of successes whereas $x_{i}$ is a single trial.

Now let's compute the MAP estimator of $\theta$. Since conjugate prior of binomial distribution is gamma distribution, we use gamma distribution to express $P(\theta)$ here. Gamma distribution is described as below:

\begin{equation}
P(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta^{\alpha -1} (1-\theta)^{\beta -1}
\end{equation}

where $\alpha$ and $\beta$ are called hyperparameters where cannot be computed from data. Rather we set them subjectively to express our prior knowledge.

\begin{equation}
\begin{split}
\theta_{MAP} &= \underset{\theta}{\operatorname{argmax}} \sum_{i=1} log \left(P(x_{i} \mid \theta)\right) + log \left( P(\theta)\right)\\
&=\sum_{i=1} \left[log {{n}\choose{x_{i}}} + x_{i} log (\theta) + (n-x_{i}) log(1-\theta)\right] + (\alpha -1) log (\theta) + (\beta -1) log (1-\theta)
\end{split}
\end{equation}

As same as MLE, we can get $\theta$ maximizing this equation by having the derivative of this function with respect to $\theta$ and setting it 0:

\begin{equation}
\theta_{MAP} = \frac{\sum_{i=1}x_{i} + \alpha -1}{\theta} - \frac{(n - \sum_{i=1}x_{i} + \beta -1)}{1-\theta} = 0
\end{equation}

gives that $\theta_{MAP} =\frac{\sum_{i=1}x_{i} + \alpha -1}{n+\alpha + \beta -2}$

# Differences

Comparing MLE and MAP equations, the only thing that differs is the inclusion of the prior distribution of $\theta$, $P(\theta)$ in MAP, otherwise they are identical. What it means that the likelihood is now weighted with some weight coming from the prior. Let's consider that we use the simplest prior in out MAP estmation, i.e., prior uniform. This means that we assign equal weights on all possible values of $\theta$. The implication is that the likelihood equivalently weighted by some constants. Being constant, we could simply ignore it from our MAP equation, as it will not contribute to the maximization.

Let's be more concrete and let's say we could assign six possible vales into $\theta$. Now our prior $P(\theta)$ is $\frac{1}{6}$ everywhere in the distribution, and consequently, we could ignore the constant in our MAP estimation.

\begin{equation}
\begin{split}
\theta_{MAP} &= \underset{\theta}{\operatorname{argmax}} \sum_{i=1} log \left(P(x_{i} \mid \theta)\right) + log \left( P(\theta)\right)\\
&=\underset{\theta}{\operatorname{argmax}} \sum_{i=1} log \left(P(x_{i} \mid \theta)\right) + \text{constant}\\
&=\underset{\theta}{\operatorname{argmax}} \sum_{i=1} log \left(P(x_{i} \mid \theta)\right) \\
&=\theta_{MLE}
\end{split}
\end{equation}

We are back at MLE equation again.

If we use a different prior, say a Gaussian, then our prior is not constant anymore, as depending on the region of the distribution, the probability is high or low, never always be the same.

Placing a nonuniform prior can be thought of as regularizing the estimation, penalizing values away from maximizing the likelihood, which can lead to overfitting. 
