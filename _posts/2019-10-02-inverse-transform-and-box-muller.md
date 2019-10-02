---
layout: post
title: "Inverse Transform Method for Continuous Distributions and Sampling from Normal Distribution"
author: "MMA"
comments: true
---
# Transformation of Random Variables

Let's consider how to take the transformation of a random variable $X$ with cumulative distribution function $F_{X}(x)$. Let $Y=t(X)$, that is, $Y$ is the transformation of $X$ via function $t(X)$.

In order to get the CDF of $Y$ we use the definition of CDFs:

$$
F_{Y}(y) = P(Y \leq y) = P(t(X) \leq y)
$$

We have $F_{X}(x)$ and want to know how to compute $F_{Y}(y)$ in terms of $F_{X}(x)$. To get there we can take the inverse of $t(x)$ on both sides of the inequality:

$$
F_{Y}(y) = P(Y \leq y) = P(t(X) \leq y) = P(X \leq t^{-1}(y))
$$

This is the CDF of $X$:

$$
P(X \leq r^{-1}(y)) = F_{X}(t^{-1}(y))
$$

and that's how we get $F_{Y}(y)$ in terms of $F_{X}(x)$. We can compute the density function $f_{Y}(y)$ by differentiating $F_{Y}(y)$, applying the chain rule:

$$
f_{Y}(y) = f_{y}(t^{-1}(y)) \times \frac{d}{dy} t^{-1}(y) dy
$$

Note that it is only this simple if $t(x)$ is one-to-one and strictly monotone increasing; it gets more complicated to reason about the regions where $Y$ is defined otherwise.

How do we use this result?

Let $U \sim U(0, 1)$. Then $F(X) = U$ means that the random variable $F^{−1}(U)$ has the same distribution as $X$.

# Inverse transform sampling

It is a basic method for pseudo-random number sampling, i.e. for generating sample numbers at random from any probability distribution given its cumulative distribution function.

The problem that the inverse transform sampling method solves is as follows:

* Let $X$ be a random variable whose distribution can be described by the cumulative distribution function $F_{X}$.
* We want to generate values of $X$ which are distributed according to this distribution.

The inverse transform sampling method works as follows:

* Generate a random number $u$ from the standard uniform distribution in the interval $[0,1]$, e.g. from $U\sim Unif [0,1]$.
* Find the inverse of the desired CDF, e.g. $F_{X}^{-1}(x)$. Inverse cumulative distribution function is also called quantile function.
* Compute $x = F_{X}^{-1}(u)$ (Solve the equation $F_{X}(x) = u$ for $X$). The computed random variable $X$ has distribution $F_{X}(x)$.

Expressed differently, given a continuous uniform variable $U$ in $[0,1]$ and an invertible cumulative distribution function $F_{X}$, the random variable $X=F_{X}^{-1}(U)$ has distribution $F_{X}$ (or, $X$ is distributed $F_{X}$).

Computationally, this method involves computing the quantile function of the distribution — in other words, computing the cumulative distribution function (CDF) of the distribution (which maps a number in the domain to a probability between 0 and 1) and then inverting that function many times. This is the source of the term "inverse" or "inversion" in most of the names for this method. Note that for a discrete distribution, computing the CDF is not in general too difficult: we simply add up the individual probabilities for the various points of the distribution. For a continuous distribution, however, we need to integrate the probability density function (PDF) of the distribution, which is impossible to do analytically for most distributions (including the normal distribution). As a result, this method may be computationally inefficient for many distributions and other methods are preferred; however, it is a useful method for building more generally applicable samplers such as those based on rejection sampling.

For the normal distribution, the lack of an analytical expression for the corresponding quantile function means that other methods (e.g. the Box–Muller transform) may be preferred computationally. It is often the case that, even for simple distributions, the inverse transform sampling method can be improved on.

(Note: technically this only works when the CDF has a closed form inverse function)

# Continuous Example: Exponential Distribution

The exponential distribution has CDF:

$$
F_X(x) = 1 - e^{-\lambda x}
$$

for $x \geq 0$ (and $0$ otherwise). By solving $u=F(x)$ we obtain the inverse function

$$
\begin{split}
1 - e^{-\lambda x} &= u\\
x &= \frac{-1}{\lambda}ln(1 - y)
\end{split}
$$

so

$$
F^{-1}_X(x) = \frac{-1}{\lambda}ln(1 - u)
$$

It means that if we draw some $u$ from $U \sim Unif(0,1)$ and compute $x = F^{-1}_X(x) = \frac{-1}{\lambda}ln(1 - u)$, this $X$ has exponential distribution.

Note that in practice, since both $u$ AND $1-u$ are uniformly distributed random number, so the calculation can be simplified as:

$$
x = F^{-1}_X(x) = \frac{-1}{\lambda}ln(u)
$$

{% highlight python %} 
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def inverse_exp_dist(lmbda=1.0):
    return (-1 / lmbda)*math.log(1 - np.random.random())

plt.hist([inverse_exp_dist() for i in range(10000)], 50)
plt.title('Samples from an exponential function')
plt.savefig('inverse_pdf_exp_dist')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/inverse_pdf_exp_dist.png?raw=true)

and just to make sure this looks right, let's use numpy's exponential function and compare:

{% highlight python %} 
plt.hist([np.random.exponential() for i in range(10000)], 50)   
plt.title('Samples from numpy.random.exponential')
plt.savefig('numpy_random_exponential_dist')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/numpy_random_exponential_dist.png?raw=true)

# Functions with no inverses

In general, there are no inverses for functions that can return same value for different inputs, for example density functions (e.g., the standard normal density function is symmetric, so it returns the same values for −2 and 2 etc.). The normal distribution is an interesting example for one more reason—it is one of the examples of cumulative distribution functions that do not have a closed-form inverse. Not every cumulative distribution function has to have a closed-form inverse! Hopefully in such cases the inverses can be found using numerical methods.

# Normal Distribution

There's no closed form expression for the inverse cdf of a normal distribution (a.k.a. the quantile function of a normal distribution). This is often a
problem with the inverse transform method. There are various ways to express the function and numerous approximations.

The standard normal distribution. Unfortunately, the inverse c.d.f. $\Phi^{-1}(\cdot)$ does not have an analytical form. This is often a problem with the inverse transform method.

We can do a table lookup. E.g., If $U = 0.975$, then $Z = \Phi^{-1}(U) = 1.96$ because z-table gives $\Phi(Z)$.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/z_table.png?raw=true)

If we are willing to accept numeric solution, inverse functions can be found. One of the inverse c.d.f. of the standard normal distribution was proposed by Schmeiser:

$$
Z = \Phi^{-1}(U) \approx \frac{U^{0.135} - (1 - U)^{0.135}} {0.1975}
$$

for $0.0013499 \le U \le 0.9986501$ which matches the true normal distribution with one digit after decimal point. 

There is one another approximation. The following approximation has absolute error $\leq 0.45 \times 10^{−3}$:

$$
Z = sign(U − 1/2) \left(t - \frac{c_{0} + c_{1}t + c_{2} t^{2}}{1 + d_{1}t + d_{2} t^{2} + d_{3}t^{3}} \right)
$$

where sign(x) = 1, 0, −1 if $X$ is positive, zero, or negative, respectively,

$$
t = \left\{- \ln \left[min (U, 1-U) \right]^{2} \right\}^{1/2}
$$

and $c_{0} = 2.515517, c_{1} = 0.802853, c_{2} = 0.010328, d_{1} = 1.432788, d_{2} = 0.189269, d_{3} = 0.001308$.

In any case, if $Z \sim N(0, 1)$ and you want $X \sim N(\mu, \sigma^{2})$, just take $X \leftarrow \mu + \sigma Z$. Suppose you want to generate $X \sim N(3, 16)$, and you start with $U = 0.59$. Then,

$$
X = \mu + \sigma Z = 3 + 4 \Phi^{-1}(0.59) = 3 + 4(0.2275) = 3.91
$$

because $\Phi^{-1}(0.59) = Z \rightarrow \Phi(Z) = P(Z \leq Z) = 0.59$. What is this $Z$? Using a [online calculator](https://stattrek.com/online-calculator/normal.aspx){:target="_blank"}, it is $0.2275$.

### The Box–Muller method 
Now let's consider a more direct transormation. Let $Z_1, Z_2$ be two standard normal random variates. Plot the two as a point in the plane and represent them in a polar coordinate system as $Z_1 = B \cos \theta$ and $Z_2 = B \sin \theta$.

It is known that $B^2 = {Z_1}^2 + {Z_2}^2$ has the chi-square distribution with 2 degrees of freedom, which is equivalent to an exponential distribution with mean 2 (this comes from the fact that if one has $k$ i.i.d normal random variables where $X_i\sim N(0,\sigma^2)$, sum of squares of those random variables, $X_1^2+X_2^2+\dots+X_k^2\sim\sigma^2\chi^2_k$):

$$
Y = \lambda e^{-\lambda t},\,\,\,\,\, t \geq 0
$$

where $E[Y] = 2 = \lambda$. Thus, the raidus $B$ can be generated using $B = \sqrt{-2\ln U}$. 

Note that here, we use alternative formulation of Exponential distribution, where:

$$
f(x) = \frac{1} {\lambda} e^{-x/\lambda},\,\,\,\,\, x \geq 0; \lambda > 0
$$

with mean $E(X) = \lambda$ and variance $Var(X)=\lambda^{2}$

$$
F(x) = 1 - e^{-x/\lambda},\,\,\,\,\, x \ge 0; \lambda > 0
$$

So, the formula for inverse of CDF (quantile function or the percent point function) of the exponential distribution is

$$
F^{-1}_{X}(x) = -\lambda\ln(1 - u)
$$

Again, in practice, since both $u$ AND $1-u$ are uniformly distributed random number.

So a standard normal distribution can be generated by any one of the following.

$$
Z_1 = \sqrt{-2 \ln U_1} \cos (2\pi U_2)
$$

and

$$
Z_2 = \sqrt{-2 \ln U_1} \sin (2\pi U_2)
$$

where $U_1$ and $U_2$ are uniformly distributed over $(0,1)$ and they will be independent. In order to obtain normal variates $X_i$ with mean $\mu$ and variance $\sigma^2$, transform $X_i = \mu + \sigma Z_i$.

However, in this computer age, most statistical software would provide you with quantile function for normal distribution already implemented. The inverse of the normal CDF is know and given by:

$$
F^{-1}(Z)\; =\; \sqrt2\;\operatorname{erf}^{-1}(2Z - 1), \quad Z\in(0,1).
$$

Hence:

$$
Z = F^{-1}(U)\; =\; \sqrt2\;\operatorname{erf}^{-1}(2U - 1), \quad U\in(0,1)
$$

where erf is error function.

#### REFERENCES

1. [https://www.quora.com/What-is-an-intuitive-explanation-of-inverse-transform-sampling-method-in-statistics-and-how-does-it-relate-to-cumulative-distribution-function/answer/Amit-Sharma-2?srid=X8V](https://www.quora.com/What-is-an-intuitive-explanation-of-inverse-transform-sampling-method-in-statistics-and-how-does-it-relate-to-cumulative-distribution-function/answer/Amit-Sharma-2?srid=X8V){:target="_blank"}
2. [http://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node48.html](http://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node48.html){:target="_blank"}
3. [http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf](http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf){:target="_blank"}
4. [http://people.duke.edu/~ccc14/sta-663-2016/15A_RandomNumbers.html](http://people.duke.edu/~ccc14/sta-663-2016/15A_RandomNumbers.html){:target="_blank"}
5. [http://karlrosaen.com/ml/notebooks/simulating-random-variables/](http://karlrosaen.com/ml/notebooks/simulating-random-variables/){:target="_blank"}
6. [https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html](https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html){:target="_blank"}
7. [https://stats.stackexchange.com/a/236157/16534](https://stats.stackexchange.com/a/236157/16534){:target="_blank"}
8. [https://www2.isye.gatech.edu/~sman/courses/6644/Module07-RandomVariateGenerationSlides_171116.pdf](https://www2.isye.gatech.edu/~sman/courses/6644/Module07-RandomVariateGenerationSlides_171116.pdf){:target="_blank"}
9. [http://bjlkeng.github.io/posts/sampling-from-a-normal-distribution/](http://bjlkeng.github.io/posts/sampling-from-a-normal-distribution/){:target="_blank"}
10. [https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution](https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution){:target="_blank"}
