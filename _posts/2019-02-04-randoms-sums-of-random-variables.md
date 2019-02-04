---
layout: post
title: "Expected value and variance of sum of a random number of i.i.d. random variables"
author: "MMA"
comments: true
---

Let $N$ be a random variable assuming positive integer values $1, 2, 3, \dots$. Let $X_{i}$ be a sequence of independent random variables which are also independent of $N$ and with common mean $E[X_{i}] = E[X]$ the same for all $i$ and common variance $Var[X_{i}] = Var[X] $the same for all $i$, meaning that they do not depend on $i$. Then,

$$
\begin{split}
E \left[\sum_{i=1}^{N} X_{i}   \right] &= E_{N} \left[ E_{X_{i}} \left[ \sum_{i=1}^{N} X_{i} \middle| N = n \right]\right]\\
&=\sum_{n = 1}^{\infty} E_{X_{i}} \left[ \sum_{i=1}^{N} X_{i} \middle| N = n \right] P(N = n) \\
&= \sum_{n = 1}^{\infty} E_{X_{i}} \left(X_{1} + X_{2} +\cdots +\X_{n} \righ) P(N = n) \\
&= \sum_{n = 1}^{\infty} E_{X_{i}} \left[\sum_{n = 1}^{n} X_{i} \right] P(N = n) \\
&=  \sum_{n = 1}^{\infty}\sum_{n = 1}^{n} E_{X_{i}}\left( X_{i} \right)P(N = n) \\
&= \sum_{n = 1}^{\infty} n E_{X_{i}}\left(X_{i} \right) P(N = n) \\
&= \left[n  P(N = n) \right] E_{X_{i}}\left(X_{i} \right)\\
&=E(N)E(X)
\end{split}
$$