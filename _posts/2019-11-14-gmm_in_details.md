---
layout: post
title: "Gaussian Mixture Models in theoretical details"
author: "MMA"
comments: true
---

NOTE: This blog post consists of images. It might take a while to load in your browser!


![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-01.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-02.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-03.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-04.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-05.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-06.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-07.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-08.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-09.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-10.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-11.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-12.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-13.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-14.png?raw=true)

{% highlight python %}
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

mu = [5, 10]
sigma = [1.5, 2]
p_i = [0.25, 0.75]
n = 10000

x = []
for i in range(n):
    z_i = np.random.binomial(1, 0.75)
    x_i = np.random.normal(mu[z_i], sigma[z_i])
    x.append(x_i)

def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))

a = np.arange(0, 18, 0.01)
y = p_i[0] * univariate_normal(a, mean=mu[0], variance=1.5**2) + p_i[1] * univariate_normal(a, mean=mu[1], variance=4)

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(x, bins='auto', density=True)
ax.plot(a, y)
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/two_gaussians.png?raw=true)

# Three Gaussians Mixture distribution

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

mu = [0, 10, 3]
sigma = [1, 1, 1]
p_i = [0.3, 0.5, 0.2]
n = 10000

x = []
for i in range(n):
    z_i = np.argmax(np.random.multinomial(1, p_i))
    x_i = np.random.normal(mu[z_i], sigma[z_i])
    x.append(x_i)
    
def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))

a = np.arange(-7, 18, 0.01)
y = p_i[0] * univariate_normal(a, mean=mu[0], variance=sigma[0]**2) + p_i[1] * univariate_normal(a, mean=mu[1], variance=sigma[0]**2)+ p_i[2] * univariate_normal(a, mean=mu[2], variance=sigma[0]**2)

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(x, bins=100, density=True)
ax.plot(a, y)
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/three_gaussians.png?raw=true)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC111319-11132019141108-15.png?raw=true)
