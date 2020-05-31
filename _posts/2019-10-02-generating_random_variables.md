---
layout: post
title: "Generating random variables"
author: "MMA"
comments: true
---
# Transformation of Random Variables

Let's consider how to take the transformation of a random variable $X$ with cumulative distribution function $F_{X}(x)$. Let $Y=t(X)$, that is, $Y$ is the transformation of $X$ via function $t(\cdot)$.

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
P(X \leq t^{-1}(y)) = F_{X}(t^{-1}(y))
$$

and that's how we get $F_{Y}(y)$ in terms of $F_{X}(x)$. We can compute the density function $f_{Y}(y)$ by differentiating $F_{Y}(y)$, applying the chain rule:

$$
f_{Y}(y) = f_{y}(t^{-1}(y)) \times \frac{d}{dy} t^{-1}(y) dy
$$

Note that it is only this simple if $t(x)$ is one-to-one and strictly monotone increasing; it gets more complicated to reason about the regions where $Y$ is defined otherwise.

How do we use this result?

Let $U \sim U(0, 1)$. Then $F(X) = U$ means that the random variable $F^{−1}(U)$ has the same distribution as $X$.

# Inverse transform sampling

It is a basic method for pseudo-random number sampling, i.e. for generating sample numbers at random from any probability distribution given its cumulative distribution function. The basic principle is to find the inverse function of F, $F^{-1}$ such that $F~ F^{-1} = F^{-1} ~ F = I$.

The problem that the inverse transform sampling method solves is as follows:

* Let $X$ be a random variable whose distribution can be described by the cumulative distribution function $F_{X}$.
* We want to generate values of $X$ which are distributed according to this distribution.

The inverse transform sampling method works as follows:

* Generate a random number $u$ from the standard uniform distribution in the interval $[0,1]$, e.g. from $U\sim Unif [0,1]$.
* Find the inverse of the desired CDF, e.g. $F_{X}^{-1}(x)$. Inverse cumulative distribution function is also called quantile function.
* Compute $x = F_{X}^{-1}(u)$ (Solve the equation $F_{X}(x) = u$ for $X$). The computed random variable $X$ has distribution $F_{X}(x)$.

Expressed differently, given a continuous uniform variable $U$ in $[0,1]$ and an invertible cumulative distribution function $F_{X}$, the random variable $X=F_{X}^{-1}(U)$ has distribution $F_{X}$ (or, $X$ is distributed $F_{X}$).

$$
\begin{split}
F_{X}(x) = P(X \leq x) &= P(F_{X}^{-1}(U)\leq x)\\
&=P(U \leq F_{X}(x))\\
&= F_{U}(F_{X}(x))\\
&= F_{X}(x) 
\end{split}
$$

Remember that the cumulative distribution function of continuous uniform distribution on the interval $[0,1]$ is $F_{U}(u)=u
$.

Computationally, this method involves computing the quantile function of the distribution — in other words, computing the cumulative distribution function (CDF) of the distribution (which maps a number in the domain to a probability between 0 and 1) and then inverting that function many times. This is the source of the term "inverse" or "inversion" in most of the names for this method. Note that for a discrete distribution, computing the CDF is not in general too difficult: we simply add up the individual probabilities for the various points of the distribution. For a continuous distribution, however, we need to integrate the probability density function (PDF) of the distribution, which is impossible to do analytically for most distributions (including the normal distribution). As a result, this method may be computationally inefficient for many distributions and other methods are preferred; however, it is a useful method for building more generally applicable samplers such as those based on rejection sampling.

For the normal distribution, the lack of an analytical expression for the corresponding quantile function means that other methods (e.g. the Box–Muller transform) may be preferred computationally. It is often the case that, even for simple distributions, the inverse transform sampling method can be improved on.

(Note: technically this only works when the CDF has a closed form inverse function)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-06%20at%2019.45.52.png?raw=true)

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

In general, there are no inverses for functions that can return same value for different inputs, for example density functions (e.g., the standard normal density function is symmetric, so it returns the same values for −2 and 2 etc.). The normal distribution is an interesting example for one more reason—it is one of the examples of cumulative distribution functions that do not have a closed-form inverse. Not every cumulative distribution function has to have a closed-form inverse! Therefore,  the inverse transform method is not efficient. Hopefully in such cases the inverses can be found using numerical methods.

# Normal Distribution

There's no closed form expression for the inverse cdf of a normal distributio (a.k.a. the quantile function of a normal distribution). This is often a problem with the inverse transform method. There are various ways to express the function and numerous approximations.

Let's think of a standard normal distribution. The drawback of using inverse CDF method is that it relies on calculation of the probit function $\Phi^{-1}$, which cannot be done analytically (Note that In probability theory and statistics, the probit function is the quantile function associated with the standard normal distribution, which is commonly denoted as N(0,1)). Some approximate methods are described in the literature. One of the easiest way is to do a table lookup. E.g., If $U = 0.975$, then $Z = \Phi^{-1}(U) = 1.96$ because z-table gives $\Phi(Z)$.

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

In any case, rather than sampling x directly, we could instead sample $Z \sim N(0, 1)$ and transform samples of $Z$ into samples of $X$. If $Z \sim N(0, 1)$ and you want $X \sim N(\mu, \sigma^{2})$, just take $X \leftarrow \mu + \sigma Z$. Suppose you want to generate $X \sim N(3, 16)$, and you start with $U = 0.59$. Then,

$$
X = \mu + \sigma Z = 3 + 4 \Phi^{-1}(0.59) = 3 + 4(0.2275) = 3.91
$$

because $\Phi^{-1}(0.59) = Z \rightarrow \Phi(Z) = P(Z \leq Z) = 0.59$. What is this $Z$? Using a [online calculator](https://stattrek.com/online-calculator/normal.aspx){:target="_blank"}, it is $0.2275$.

Let's see an example in Python. 

{% highlight python %} 
n = 10000  # Samples to draw
mean = 3
variance = 16
Z = np.random.normal(loc=0, scale=1.0, size=(n,))
X = mean + (np.sqrt(variance) * Z)

print(np.mean(X))
#3.0017206638273097

print(np.std(X))
#4.022342597707669

count, bins, ignored = plt.hist(X, 30, normed=True)
plt.plot(bins, univariate_normal(bins, mean, variance),linewidth=2, color='r')
plt.savefig('generated_normal_dist.png')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/generated_normal_dist.png?raw=true)

### The Box–Muller method 
Now let's consider a more direct and exact transformation. Let $Z_1, Z_2$ be two standard normal random variates. Plot the two as a point in the plane and represent them in a polar coordinate system as $Z_1 = B \cos \theta$ and $Z_2 = B \sin \theta$.

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

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

# uniformly distributed values between 0 and 1
u1 = np.random.rand(1000)
u2 = np.random.rand(1000)

# transformation function
def box_muller(u1,u2):
    z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
    z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
    return z1,z2

# Run the transformation
z1 = box_muller(u1, u2)
z2 = box_muller(u1, u2)

# plotting the values before and after the transformation
plt.figure(figsize = (20, 10))
plt.subplot(221) # the first row of graphs
plt.hist(u1)     # contains the histograms of u1 and u2 
plt.subplot(222)
plt.hist(u2)
plt.subplot(223) # the second contains
plt.hist(z1)     # the histograms of z1 and z2
plt.subplot(224)
plt.hist(z2)
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/box_muller.png?raw=true)

The Box-Muller method is not the fastest way to generate $N(0, 1)$ random variables, and numerical computing environments don't always use it. There is some cost in computing cos, sin, log and sqrt that, with clever programming can be avoided. Box-Muller remains very popular because it is simple to use.

There is also The Marsaglia polar method which is a modification of the Box–Muller method which does not require computation of the sine and cosine functions

However, in this computer age, most statistical software would provide you with quantile function for normal distribution already implemented. The inverse of the normal CDF is know and given by:

$$
F^{-1}(Z)\; =\; \sqrt2\;\operatorname{erf}^{-1}(2Z - 1), \quad Z\in(0,1).
$$

Hence:

$$
Z = F^{-1}(U)\; =\; \sqrt2\;\operatorname{erf}^{-1}(2U - 1), \quad U\in(0,1)
$$

where erf is error function.

# Characterization method (Convolution Method)

This method is another approach to sample from a distribution. In some cases, $X$ can be expressed as a sum of independent random variables $Y_{1}, Y_{2}, \ldots , Y_{n}$ where $Y_{j}$'s are iid and n is fixed and finite:

$$
X = Y_{1} + Y_{2} + \ldots + Y_{n}
$$

called n-fold convolution of distribution $Y_{j}$. Here, $Y_{j}$'s are generated more easilt.

**Algorithm**:
* Generate independent $Y_{1}, Y_{2}, \ldots , Y_{n}$ each with distribution function $F_{Y}(y)$ using the inverse transform method.
* Return $X = Y_{1} + Y_{2} + \ldots + Y_{n}$.

For example, an Erlang random variable $X$ with parameters $(n, \lambda)$ can be shown to be the sum of $n$ independent exponential random variables $Y_{i}, i=1,2, \ldots ,n$, each having a mean of $\frac{1}{\lambda}$.

$$
X = \sum_{i=1}^{n} Y_{i}
$$

Using inverse CDF method that can generate an exponential variable, an Erlang variate can be generated:

$$
X = \sum_{i=1}^{n}  \frac{-1}{\lambda}ln(u_{i}) = \frac{-1}{\lambda} ln \left(\prod_{i=1}^{n} u_{i} \right)
$$

Other examples:

* If $X_{1}, \ldtos , X_{n}$ are i.i.d. Geometric(p), then $\sum_{i=1}^{n} X_{i} \sim NegBin(n, p)$
* If $X_{1}, \ldtos , X_{n}$ are i.i.d. Normal(0,1), then $\sum_{i=1}^{n} X_{i} \sim \chi_{n}^{2}$
* If $X_{1}, \ldtos , X_{n}$ are i.i.d. Bernoulli(p), then $\sum_{i=1}^{n} X_{i} \sim Binomial(n, p)$

# Composition Method

This method applies when the distribution function $F$ can be expressed as a mixture of other distribution functions $F_{1}, F_{2}, \ldots$:

$$
F(x) = \sum_{j=1}^{\infty} p_{j}F_{j}(x),
$$

where $p_{j} \geq 0$ and $\sum_{j=1}^{\infty} p_{j} =1$, meaning that the $p_{j}$ form a discrete probability distribution

Equivalently, we can decompose the density function $f(x)$ or mass function $p(x)$ into convex combination of other density or mass functions. This method is useful if it is easier to sample from $F_{j}$'s than from $F$.

**Algorithm**:
* Generate a discrete random variable $j$ such that $P(J = j) = p_{j}$.
* Return $X$ with CDF $F_{J}(x)$ (given $J=j$, $x$ is generated independent of $J$).

For fixed $x$:
$$
\begin{split}
P(X \leq x) &= \sum_{j} P(X \leq x \mid J = j)P(J = j) \text{ (condition on } J=j\text{)}\\
&= \sum_{j} P(X \leq x \mid J = j)p_{j}\text{ (distribution of J)}\\
&= \sum_{j} F_{j}(x)p_{j} \text{ (given } J = j, X \sim F_{j}\text{)}\\
&=F_{X}(x) \text{ (decomposition of F)}
\end{split}
$$

The trick is to find $F_{j}$’s from which generation is easy and fast.

# Acceptance-Rejection Method

The majority of CDFs cannot be inverted efficiently. In other words, finding an explicit formula for $F^{−1}(y)$ for the cdf of a random variable $X$ we wish to generate, $F(x) = P(X \leq x)$, is not always possible. Moreover, even if it is, there may be alternative methods for generating a random variable, distributed as $F$ that is more efficient than the inverse transform method or other methods we have come across.

Rejection Sampling is one of the simplest sampling algorithm.

We start by assuming that the $F$ we wish to simulate from has a probability density function $f(x)$ (we cannot easily sample from); that can be either continuous or discrete distribution. 

The basic idea is to find an alternative probability distribution $G$, with density function $g(x)$ (like a Normal distribution or perhaps a  t-distribution), from which we already have an efficient algorithm for generating from (because there’s a built in function or someone else wrote a nice function), but also such that the function $g(x)$ is "close" to $f(x)$. In other words, we assume there is another density $g(x)$ and a constant $c$ such that $f(x) \leq cg(x)$. Then we can sample from $g$ directly and then "reject" the samples in a strategic way to make the resulting "non-rejected" samples look like they came from $f$. The density $g$ will be referred to as the "candidate density" and $f$ will be the "target density".

In particular, we assume that the ratio $f(x)/g(x)$ is bounded by a constant $c > 0$; $sup_{x}\{f(x)/g(x)\} \leq c$. (And
in practice we would want $c$ as close to 1 as possible). The easiest way to satisfy this assumption is to make sure that  
$g$ has heavier tails than $f$. We cannot have that $g$ decreases at a faster rate than $f$ in the tails or else rejection sampling will not work.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/acception_rejection_algorithm.png?raw=true)

Here is the rejection sampling algorithm for drawing a sample from the target density $f$ is then:

1. Generate a random variable $Y$, distributed as $G$.
2. Generate $U \sim Uniform(0, 1)$ (independent from $Y$).
3. If
  $$
  U\leq\frac{f(Y)}{c\,g(Y)}
  $$
  then set $X = Y$ (*accept*) ; otherwise go back to 1 (*reject*).

The algorithm can be repeated until the desired number of samples from the target density $f$ has been accepted.

Some notes:
* $f(Y)$ and $g(Y)$ are random variables, hence, so is the ratio $\frac{f(Y)}{c\,g(Y)}$ and this ratio is independent of $U$ in Step (2).
* The ratio is bounded between 0 and 1; $0 < \frac{f(Y)}{c\,g(Y)} \leq 1$.
* The number of times $N$ that steps 1 and 2 need to be called (e.g., the number of iterations needed to successfully generate X ) is itself a random variable and has a geometric distribution with "success" probability $p = P(U\leq\frac{f(Y)}{c\,g(Y)})$. $P(N = n) = (1−p)^{n−1} p, \,\,\, n \geq 1$. Thus on average the number of iterations required is given by $E(N) = \frac{1}{p}$.
* In the end we obtain our $X$ as having the conditional distribution of a $Y$ given that the event $U \leq \frac{f(Y)}{cg(Y)}$ occurs.

A direct calculation yields that $p = \frac{1}{c}$, by first conditioning on Y, $P(U\leq\frac{f(Y)}{c\,g(Y)} \mid Y = y) = \frac{f(y)}{c\,g(y)}$, thus, unconditioning and recalling that $Y$ has density $g(y)$ yields

$$
\begin{split}
p = \int_{- \infty}^{+ \infty} \frac{f(y)}{c\,g(y)} \times g(y) \times dy\\
&= \frac{1}{c} \int_{- \infty}^{+ \infty} f(y)d(y)\\
&= \frac{1}{c}
\end{split}
$$

where the last equality follows since f is a density function (hence by definition integrates to 1).  Thus $E(N) = c$, the bounding constant, and we can now indeed see that it is desirable to choose our alternative density g so as to minimize this constant $c = sup_{x}\{f(x)/g(x)\}$. Of course the optimal function would be $g(x) = f(x)$ which is not what we have in mind since the whole point is to choose a different (easy to simulate) alternative from $f$. In short, it is a bit of an art to find an appropriate $g$.

There are two main problems with this method. The first major problem is that if distributions are chosen poorly, like if $f(x)$ is not remotely related to $g(x)$, a lot of samples may be generated and tossed away, wasting computation cycles (as an example, if the enveloping function $cg(x)$ is considerably higher than $f(x)$ at all points, the algorithm will reject most attempted draws, which implies that an incredible number of draws may need to be made before finding a single value from $f(x)$). It may also be difficult to find an envelope with values that are greater at all points of support for the density of interest. Consider trying to use a uniform density as an envelope for sampling from a normal density. The domain of $x$ for the normal density runs from $-\infty$ to $+ \infty$, but there is no corresponding uniform density. In the limit, a $U(-\infty, +\infty)$ density would have an infinitely low height, which would make $g(x)$ fall below $f(x)$ in the center of the distribution, regardless of the constant multiple $c$ chosen. Another trouble is that a lot of samples may be taken in a specific area, getting us a lot of unwanted samples. The choices of $c$ and $g$ affect the computational efficiency of the algorithm. In the case of multidimensional random vectors, we have high chance of running straight into the curse of dimensionality, where chances are corners and edges of our multidimensional density simply don't get the coverage we were hoping for.

For example, let's try to simulate random normal variates using Gamma distribution. Let the target distribution, $f(x)$ be a normal distribution with a mean of 4.5 and a standard deviation of 1. Let's choose a candidate distribution as Gamma distribution with a mean of 4 and a standard deviation 2, which results in parameters shape = 4 and scale = 1 (There is no particular reason to use the gamma distribution here – it was chosen primarily to distinguish it from the normal target distribution). Though theoretically the normal distribution extends from $-\infty$ to $\infty$ and the gamma distribution
extends from $0$ to $\infty$, it is reasonable here to only consider values of $x$ between $0$ and $13$. We choose $c = 3$ here to blanket over the target distribution. The target distribution, candidate distribution and blanket distribution (also known as envelope function) were shown below:

```python
import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns

mu = 4.5; # Mean for normal distribution
sigma = 1; # standard deviation for normal distribution
shape = 4 # Shape parameter for gamma distributiom
scale = 1 # Rate parameter for gamma distribution

# Choose value for c
c = 3

# Define x axis
x = np.arange(start = 0, stop = 12, step = 0.01)
plt.figure(num=1, figsize = (20, 10))
# Plot target distribution
plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), lw=2, label=' Target Distribution - Normal Distribution')
# Plot candidate distribution
plt.plot(x, gamma.pdf(x, a = shape, loc=0, scale=scale), lw=2, label = 'Candidate Distribution - Gamma Distribution')
# Plot the blanket function
plt.plot(x, c * gamma.pdf(x, a = shape, loc=0, scale=scale), lw=2, label = 'Blanket function - c * Gamma Distribution')
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend(loc="upper right")
plt.savefig('target_candidate_blanket_dists.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/target_candidate_blanket_dists.png?raw=true)

Having verified that the blanket function satisfies $f(x) \leq cg(x)$, the sampling process can begin.

```python
# Choose number of values desired (this many values will be accepted)
N = 20000;

accept = []
reject = []
i = 0
j = 0
while i < N:
    Y = gamma.rvs(a =shape, loc=0, scale=scale, random_state=None)
    U = np.random.rand(1)
    if U * c * gamma.pdf(Y, a = shape, loc=0, scale=scale) <= norm.pdf(Y, loc=mu, scale=sigma):
        accept.append(Y)
        i += 1
    else:
        reject.append(Y)
        j += 1
    
#PLOT RESULTS OF SAMPLING
plt.figure(num = 2, figsize = (20, 10))
plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), lw=2, label=' Target Distribution - Normal Distribution')
plt.hist(accept, bins = 40, density=True)
plt.savefig('accept_reject_algo_example.png')
plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/accept_reject_algo_example.png?raw=true)

This sampling method is performed until 20,000 values of $x$ were accepted. By inspection, the histogram of values sampled
from $f(x)$ reproduce $f(x)$ demonstrating that the rejection method successfully drew random samples from the target distribution $f$. The mean and standard deviation of sampled data points are given below:

```python
print(np.mean(accept))
#4.50210853818239
print(np.std(accept))
#0.9957279285265107
```

# Gibbs Sampling

Gibbs sampling, proposed in the early 1990s, is commonly used as a means of statistical inference, especially Bayesian inference. It is a the most basic randomized algorithm (i.e. an algorithm that makes use of random numbers), and is an alternative to deterministic algorithms for statistical inference such as the expectation-maximization algorithm (EM). It is a very useful way of simulating from distributions that are difficult to simulate from directly.

Gibbs sampling is attractive because it can sample from high-dimensional posteriors. The main idea is to break the problem of sampling from the high-dimensional joint distribution into a series of samples from low-dimensional conditional distributions. Because the low-dimensional updates are done in a loop, samples are not independent as in rejection sampling. The dependence of the samples turns out to follow a Markov distribution, leading to the name Markov chain Monte Carlo (MCMC).

The algorithm begins by setting initial values for all parameters, $\mathbf{\theta}^{(0)} = (\theta_{1}^{(0)}, \theta_{2}^{(0)}, \ldots, \theta_{p}^{(0)})$. The initial values of the variables can be determined randomly or by some other algorithm such as expectation-maximization. Variables are then sampled one at a time from their full conditional distribution

$$
p\left( \theta_{j}  \mid \theta_{1}, ..., \theta_{j-1}, \theta_{j+1}, ..., \theta_{p}, \mathbf{y} \right)
$$

Rather than 1 sample from $p$-dimensional joint, we make $p$ 1-dimensional samples. The process is repeated until the required number of samples have been generated. It is common to ignore some number of samples at the beginning (the so-called burn-in period). Formally, the algorithm is:

1. Initialize $\mathbf{\theta}^{(0)} = (\theta_{1}^{(0)}, \theta_{2}^{(0)}, \ldots, \theta_{p}^{(0)})$
2. for $j = 1, 2, \ldots$ do:
  $$
  \begin{split}
  \theta_{1}^{(j)} &\sim P(\theta_{1}^{(j)} \mid \theta_{2}^{(j - 1)}, \theta_{3}^{(j - 1)}, \ldots , \theta_{p}^{(j - 1)})\\
  \theta_{2}^{(j)} &\sim P(\theta_{2}^{(j)} \mid \theta_{1}^{(j - 1)}, \theta_{3}^{(j - 1)}, \ldots , \theta_{p}^{(j - 1)})\\
  & \ldots \ldots \ldots \\
  \theta_{p}^{(j)} &\sim P(\theta_{p}^{(j)} \mid \theta_{1}^{(j - 1)}, \theta_{2}^{(j - 1)}, \ldots , \theta_{p-1}^{(j - 1)})\\
  \end{split}
  $$
3. end for

In other words, Gibbs sampling involves ordering the parameters and sampling from the conditional distribution for each parameter given the current value of all the other parameters and repeatedly cycling through this updating process. Each "loop" through these steps is called an “iteration” of the Gibbs sampler, and when a new sampled value of a parameter is obtained, it is called an "updated" value.


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
11. [https://web.ics.purdue.edu/~hwan/IE680/Lectures/Chap08Slides.pdf](https://web.ics.purdue.edu/~hwan/IE680/Lectures/Chap08Slides.pdf){:target="_blank"}
12. [https://www.win.tue.nl/~marko/2WB05/lecture8.pdf](https://www.win.tue.nl/~marko/2WB05/lecture8.pdf){:target="_blank"}
13. [http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-ARM.pdf](http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-ARM.pdf){:target="_blank"}
14. [http://statweb.stanford.edu/~owen/mc/Ch-nonunifrng.pdf](http://statweb.stanford.edu/~owen/mc/Ch-nonunifrng.pdf){:target="_blank"}
