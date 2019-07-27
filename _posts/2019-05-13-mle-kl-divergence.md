---
layout: post
title: "Relation between Maximum Likelihood and KL-Divergence"
author: "MMA"
comments: true
---

Maximizing likelihood is equivalent to minimizing KL-Divergence. Considering $P( \cdot \mid \theta^{*})$ is the true distribution and $P(\cdot \mid \theta)$ is our estimate, we already know that KL-Divergence is written as:

$$
\begin{split}
D_{KL} \left[P(x | \theta^{*}) ||   P(x | \theta)\right] & = E_{x {\sim} P(x | \theta^{*})}
\left[\log \frac{P(x | \theta^{*}) }{P(x | \theta)} \right]\\
& =  E_{x {\sim} P(x | \theta^{*})} \left[\log P(x | \theta^{*}) - \log P(x | \theta)   \right]\\
& = E_{x {\sim} P(x | \theta^{*})}\left[ \log P(x | \theta^{*}) \right] - E_{x {\sim} P(x | \theta^{*})} \left[ \log P(x | \theta) \right]\\
\end{split}
$$

On the right side of the equation, the first term is the entropy of the true distribution. It does not depend on the estimated parameter $\theta$ so we can ignore it. 

Suppose we have n observations of the distribution $x \sim P(x | \theta^{*})$. Then, Law of Large Numbers says that as 
$n$ goes to infinity,

$$
-\frac{1}{n} \sum_{i=1}^{n} \log P(x_{i} | \theta) = - E_{x {\sim} P(x | \theta^{*})} \left[ \log P(x | \theta) \right]
$$

which gives the second term of above KL divergence equation. Notice that $-\sum_{i=1}^{n} \log P(x_{i} \mid \theta)$ is the negative log-likelihood of a distribution. Then if we minimize $D_{KL} \left[P(x \mid \theta^{*}) \mid\mid  P(x \mid \theta)\right]$, it is equivalent to minimizing the negative log-likelihood, in other words, it is equivalent to maximizing log-likelihood. This is important because this gives MLE a nice interpretation: maximizing the likelihood of data under our estimate is equal to minimizing the difference between our estimate and the real data distribution. We can see MLE as a proxy for fitting our estimate to the real distribution, which cannot be done directly as the real distribution is unknown to us.
