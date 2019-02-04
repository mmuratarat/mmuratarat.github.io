---
layout: post
title: "Expected value and variance of sum of a random number of i.i.d. random variables"
author: "MMA"
comments: true
---

Let $N$ be a random variable assuming positive integer values $1, 2, 3, \dots$. Let $X_{i}$ be a sequence of independent random variables which are also independent of $N$ and with common mean $E[X_{i}] = E[X]$ the same for all $i$ and common variance $Var[X_{i}] = Var[X]$ the same for all $i$, meaning that they do not depend on $i$. Then,

$$
\begin{split}
E \left[\sum_{i=1}^{N} X_{i}   \right] &= E_{N} \left[ E_{X_{i}} \left[ \sum_{i=1}^{N} X_{i} \middle| N = n \right]\right]\\
&=\sum_{n = 1}^{\infty} E_{X_{i}} \left[ \sum_{i=1}^{N} X_{i} \middle| N = n \right] P(N = n) \\
&= \sum_{n = 1}^{\infty} E_{X_{i}} \left(X_{1} + X_{2} + \cdots + X_{n} \right) P(N = n) \\
&= \sum_{n = 1}^{\infty} E_{X_{i}} \left[\sum_{i = 1}^{n} X_{i} \right] P(N = n) \\
&=  \sum_{n = 1}^{\infty}\sum_{i = 1}^{n} E_{X_{i}}\left( X_{i} \right)P(N = n) \\
&= \sum_{n = 1}^{\infty} n E_{X_{i}}\left(X_{i} \right) P(N = n) \\
&= \left[\sum_{n = 1}^{\infty} n  P(N = n) \right] E_{X_{i}}\left(X_{i} \right)\\
&=E(N)E(X)
\end{split}
$$

After computing expected value, we can now compute the variance. We know that $Var(Y) = E(Y^{2}) - (E(Y))^{2}$. Then,

$$
Var \left[\sum_{i=1}^{N} X_{i} \right] = E \left[\left(\sum_{i=1}^{N} X_{i}\right)^{2}\right] - \left( E \left[\sum_{i=1}^{N} X_{i}   \right]\right)^{2}
$$

We know how to evaluate the second term. Let’s work on the first term. Condition on N.

$$
E \left[\left(\sum_{i=1}^{N} X_{i}\right)^{2}\right] =  E_{N} \left[E_{X_{i}} \left[\left(\sum_{i=1}^{N} X_{i}\right)^{2}\middle| N = n\right]\right]
$$

Now let's work on the innermost conditional expectation:

$$
\begin{split}
E_{X_{i}} \left[\left(\sum_{i=1}^{N} X_{i}\right)^{2}\middle| N = n\right] &= E_{X_{i}} \left[\left(\sum_{i=1}^{n} X_{i}\right)^{2}\right]\\
&= Var \left[\sum_{i=1}^{n} X_{i} \right] + \left[E_{X_{i}} \left[\sum_{i=1}^{n} X_{i} \right]  \right]^{2}\\
&=nVar(X_{i}) + \left[n E(X_{i}) \right]^{2}\\
&=nVar(X) + \left[n E(X) \right]^{2}
\end{split}
$$

Now perform the outer expectation with respect to N.

$$
\begin{split}
E_{N} \left[E_{X_{i}} \left[\left(\sum_{i=1}^{N} X_{i}\right)^{2}\middle| N = n\right]\right] &=E_{N} \left[nVar(X) + \left[n E(X) \right]^{2}\right]\\
&= E_{N} \left[nVar(X) + n^{2} E(X)^{2}\right]\\
&=\sum_{n=1}^{\infty} \left[nVar(X) + n^{2} E(X)^{2}\right] P(N = n)\\
&=E[N]Var[X] + E\left[N^{2} \right] E[X]^{2}
\end{split}
$$

Therefore,

$$
\begin{split}
Var \left[\sum_{i=1}^{N} X_{i} \right]  &=E \left[\left(\sum_{i=1}^{N} X_{i}\right)^{2}\right] - \left( E \left[\sum_{i=1}^{N} X_{i}   \right]\right)^{2}\\
&= E[N]Var[X] + E\left[N^{2} \right] E[X]^{2} - \left[E(N)E(X) \right]^{2}\\
&=E[N]Var[X] + (E[X])^{2}Var[N]
\end{split}
$$
