---
layout: post
title: "an Unbiased Estimator and its proof"
author: "MMA"
comments: true
---

Unbiasness is one of the properties of an estimator in Statistics. If the following holds, where $\hat{\theta}$ is the estimate of the true population parameter $\theta$:

$$
E(\hat{\theta}) = \theta
$$

then the statistic $\hat{\theta}$ is unbiased estimator of the parameter $\theta$. Otherwise, $\hat{\theta}$ is the biased estimator.

In essence, we take the expected value of $\hat{\theta}$, we take multiple samples from the true population and compute the average of all possible sample statistics.

For exampke, if $X_{i}$ is a Bernoulli random variable with a parameter $p$, then, finding maximum likelihood estimation of the parameter $p$ of Bernoulli distribution is trivial. 

$$
L(p;X) = \prod\limits_{i=1}^n p(X_i;p) = \prod\limits_{i=1}^n p^{X_{i}}(1-p)^{1-X_{i}}
$$

Differentiating the log of $L(p;X)$ with respect to $p$ and setting the derivative to zero shows that this function achieves a maximum at $\hat{p} = \frac{\sum_{i=1}^{n} X_{i}}{n}$.

Let's find out the maximum likelihood estimator of $p$ is an unbiased estimator of $p$ or not. 

Since $X_{i} \sim Bernoulli(p)$, we know that $E(X_{i}) = p,\,\, i=1,2, \ldots , n$. Therefore,

$$
E(\hat{p}) =  E \left(\frac{\sum_{i=1}^{n} X_{i}}{n} \right) = \frac{1}{n} \sum_{i=1}^{n} E(X_{i}) = \frac{1}{n}\sum_{i=1}^{n}p = \frac{1}{n} np = p
$$

Therefore, we can safely say that the maximum likelihood estimator is an unbiased estimator of $p$.

However, this is not always the true for some other estimates of population parameters. In statistics, Bessel's correction is the use of $n-1$ instead of $n$ in the formula for the sample variance where $n$ is the number of observations in a sample. 

This method corrects the bias in the estimation of the population variance. It also partially corrects the bias in the estimation of the population standard deviation. However, the correction often increases the mean squared error in these estimations.

In the estimating population variance from a sample where population mean is unknown, the uncorrected sample variance is the mean of the squares of the deviations of sample values from the sample mean (i.e., using a multiplicative factor $\frac{1}{n}$). In this case, the sample variance is a biased estimator of the population variance.

Multiplying the uncorrected sample variance by the factor $\frac{n}{n-1}$ gives the unbiased estimator of the population variance. In some literature, the above factor is called Bessel's correction.

Let $X_{1}, X_{2}, \ldots, X_{n}$ be an i.i.d. random variables, each with the expected value $\mu$ and variance $\sigma^{2}$. For the entire population, $\sigma^{2} = E\left((X_{i} -\mu)^{2} \right)$.

When we sample from this population, we want a statistic such that $E(s^{2}) = \sigma^{2}$. Intuitively, we would guess $s^{2} = \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n}$ where $\bar{X}$ is the mean of the sample, $\bar{X} = \frac{\sum_{i=1}^{n} X_{i}}{n}$.

$$
\begin{split}
E(s^{2}) = E\left(\frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n} \right) &= \frac{1}{n}  E\left(\sum_{i=1}^{n} (X_{i} - \bar{X})^{2} \right)\\
&= \frac{1}{n}  E\left[ \sum_{i=1}^{n} \left((X_{i} - \mu) - (\bar{X} - \mu) \right)^{2} \right]\\
&=\frac{1}{n}  E\left[ \sum_{i=1}^{n} (X_{i} - \mu)^{2} - 2 \sum_{i=1}^{n} (X_{i} - \mu)(\bar{X} - \mu) + \sum_{i=1}^{n}(\bar{X} - \mu)^{2}  \right]\\
&= \frac{1}{n} \left[ \sum_{i=1}^{n} E (X_{i} - \mu)^{2} - n E (\bar{X} - \mu)^{2}  \right]\\
\end{split}
$$

Substituting $\sigma^{2} = E(X_{i} - \mu)^{2}$ and $E (\bar{X} - \mu)^{2} = \frac{\sigma^{2}}{n}$ (from central limit theorem) results in the following:

$$
\begin{split}
E(s^{2}) &= \frac{1}{n} \left(\sum_{i=1}^{n} \sigma^{2} - n \frac{\sigma^{2}}{n} \right)\\
&= \frac{1}{n} \left(n \sigma^{2} - \sigma^{2} \right) \\
&=\frac{n-1}{n} \sigma^{2}
\end{split}
$$

Thus, sample variance $s^{2}$ is a biased estimate of $\sigma^{2}$. Therefore, if we multiple both sides of the equation with $\frac{n}{n-1}$ will do the job.

$$
\frac{n}{n-1} E\left(s^{2}\right) = E\left(\frac{n}{n-1} s^{2}\right) = E\left(\frac{n}{n-1} \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n}\right) = E\left(\frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}\right) = \sigma^{2}
$$

$s^{2} = \frac{\sum_{i=1}^{n} (X_{i} - \bar{X})^{2}}{n-1}$ is the statistic that is always an unbiased estimator of the desired population parameter $\sigma^{2}$. However note that $s$ is not an unbiased estimator of $\sigma$.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator2.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator3.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/unbiased_estimator4.png?raw=true)
