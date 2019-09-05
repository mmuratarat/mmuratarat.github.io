---
layout: post
title: "Regularized Linear Models"
author: "MMA"
comments: true
---

An important distinction between linear and logistic regression is that the regression coefficients in logistic regression are not directly meaningful. In linear regression, a coefficient $\theta_{j} = 1$ means that if you change $x_{j}$ by 1, the expected value of y will go up by 1 (very interpretable). In logistic regression, a coefficient $\theta_{j} = 1$ means that if you change $x_{j}$ by 1, the log of the odds that y occus will go up 1 (much less interpretable).

# Overview of Logistic Regression
 In the linear regression model, we have modelled the relationship between outcome and features with a linear equation:
 
$$
\hat{y}^{(i)}=\theta_{0}+\theta_{1}x^{(i)}_{1}+\ldots+\theta_{p}x^{(i)}_{p}
$$

For classification, we prefer probabilities between 0 and 1, so we wrap the right side of the equation into the logistic function. This forces the output to assume only values between 0 and 1.

$$
P(y^{(i)}=1 | x^{(i), \theta)=\frac{1}{1+exp(-(\theta_{0}+\theta_{1}x^{(i)}_{1}+\ldots+\theta_{p}x^{(i)}_{p}))}
$$

The interpretation of the weights in logistic regression differs from the interpretation of the weights in linear regression, since the outcome in logistic regression is a probability between 0 and 1. The weights do not influence the probability linearly any longer. The weighted sum is transformed by the logistic function to a probability. Therefore we need to reformulate the equation for the interpretation so that only the linear term is on the right side of the formula.

$$
logit(P(y=1)) = log\left(\frac{P(y=1)}{1-P(y=1)}\right)=log\left(\frac{P(y=1)}{P(y=0)}\right)=\theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

We call the term in the log() function “odds” (probability of event divided by probability of no event) and wrapped in the logarithm it is called log odds. This formula shows that the logistic regression model is a linear model for the log odds. In other words, logistic regression models the logit transformed probability as a linear relationship with the predictor variables.

Then we compare what happens when we increase one of the feature values by 1. But instead of looking at the difference, we look at the ratio of the two predictions:

$$
\frac{odds_{x_j+1}}{odds}=\frac{exp\left(\theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{j}(x_{j}+1)+\ldots+\theta_{p}x_{p}\right)}{exp\left(\theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{j}x_{j}+\ldots+\theta_{p}x_{p}\right)}
$$

We apply the following rule:

$$
\frac{exp(a)}{exp(b)}=exp(a-b)
$$

And we remove many terms:

$$
\frac{odds_{x_j+1}}{odds}=exp\left(\theta_{j}(x_{j}+1)-\theta_{j}x_{j}\right)=exp\left(\theta_j\right)
$$

In the end, we have something as simple as exp() of a feature weight. A change in a feature by one unit changes the odds ratio (multiplicative) by a factor of $exp\left(\theta_j\right)$. We could also interpret it this way: A change in $x_{j}$ by one unit increases the log odds ratio by the value of the corresponding weight.

Interpretation of intercept term $\theta_{0}$ is a bit different. When all numerical features are zero and the categorical features are at the reference category, the estimated odds are $exp\left(\theta_{0}\right)$. The interpretation of the intercept weight is usually not relevant.

Note that probability ranges from $0$ to $1$. Odds range from $0$ to $\infty$. Log-odds range from $-\infty$ to $\infty$.

## From probability to odds/odds ratio

[https://thestatsgeek.com/2015/01/03/interpreting-odds-and-odds-ratios/](Source){:target="_blank"}

Our starting point is that of using probability to express the chance that an event of interest occurs. So a probability of $0.1$, or $10\%$ risk, means that there is a $1$ in $10$ chance of the event occurring. The usual way of thinking about probability is that if we could repeat the experiment or process under consideration a large number of times, the fraction of experiments where the event occurs should be close to the probability (e.g. $0.1$). The odds of an event of interest occurring is defined by $odds = \dfrac{p}{(1-p)}$ where $p$ is the probability of the event occurring. So if $p=0.1$, the odds are equal to $0.1/0.9=0.111$ (recurring). 

Particularly in the world of gambling, odds are sometimes expressed as fractions, in order to ease mental calculations. For example, odds of 9 to 1 against, said as "nine to one against", and written as 9/1 or 9:1, means the event of interest will occur once for every 9 times that the event does not occur. That is in 10 times/replications, we expect the event of interest to happen once and the event not to happen in the other 9 times.

In the statistics world odds ratios are frequently used to express the relative chance of an event happening under two different conditions. For example, in the context of a clinical trial comparing an existing treatment to a new treatment, we may compare the odds of experiencing a bad outcome if a patient takes the new treatment to the odds of a experiencing a bad outcome if a patient takes the existing treatment.

Suppose that the probability of a bad outcome is $0.2$ if a patient takes the existing treatment, but that this is reduced to $0.1$ if they take the new treatment. The odds of a bad outcome with the existing treatment is $0.2/0.8=0.25$, while the odds on the new treatment are $0.1/0.9=0.111$ (recurring). The odds ratio comparing the new treatment to the old treatment is then simply the correspond ratio of odds: $(0.1/0.9) / (0.2/0.8) = 0.111 / 0.25 = 0.444$ (recurring). This means that the odds of a bad outcome if a patient takes the new treatment are $0.444$ that of the odds of a bad outcome if they take the existing treatment. The odds (and hence probability) of a bad outcome are reduced by taking the new treatment. We could also express the reduction by saying that the odds are reduced by approximately $56\%$, since the odds are reduced by a factor of $0.444$.

People often find odds, and consequently also an odds ratio, difficult to intuitively interpret. An alternative is to calculate risk or probability ratios. In the clinical trial example, the risk ratio is simply the ratio of the probability of a bad outcome under the new treatment to the probability under the existing treatment, i.e. $0.1/0.2=0.5$. This means the risk of a bad outcome with the new treatment is half that under the existing treatment, or alternatively the risk is reduced by a half. Intuitively the risk ratio is much easier to understand. 

# An Example

{% highlight python %}
import pandas as pd
import numpy as np
# Needed to run the logistic regression
import statsmodels.formula.api as smf
{% endhighlight %}

Data used in this example is the data set that is used in UCLA’s Logistic Regression for Stata example. The question being asked is, how does GRE score, GPA, and prestige of the undergraduate institution effect admission into graduate school. The response (target/dependent) variable is admission status (binary), and the independent variables (predictors/features/attributes) are: GRE score, GPA, and undergraduate prestige.

{% highlight python %}
data = pd.read_csv('odds_logistic_regression.csv')
## Converting variable to categorical data type (since that what it is)
## and then creating dummy variables
data['rank'] = data['rank'].astype('category')
model= smf.logit(formula="admit~ gre + gpa + C(rank)", data=data).fit()
model.summary()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/logistic_regression_odds1.png?raw=true)

{% highlight python %}
# GETTING THE ODDS RATIOS, Z-VALUE, AND 95% CI
model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
model_odds['z-value']= model.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
model_odds
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/logistic_regression_odds2.png?raw=true)

1. For every one unit increase in gpa, the odds of being admitted increases by a factor of 2.235; for every one unit increase in gre score, the odds of being admitted increases by a factor of 1.002.
2. Still interpreting the results in comparison to the group that was dropped. Applicants from a Rank 2 University compared to a Rank 1 University are 0.509 as likely to be admitted; applicants from a Rank 3 University compared to a Rank 1 University are 0.262 as likely to be admitted, etc. An even easier way to say the above would be, applicants from a Rank 2 University are about half as likely to be admitted compared to applicants from a Rank 1 University, and applicants from a Rank 3 University are about a quarter as likely to be admitted compared to applicants from a Rank 1 University.