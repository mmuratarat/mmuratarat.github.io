---
layout: post
title: "Basic Methods to Handle Missing Data"
author: "MMA"
comments: true
---

In some cases, the data comes in the form of a dataset with features already defined. In some cases, values of some features can be missing. That often happens when the dataset was handcrafted and the person working on it forgot to fill some values or did not get them measured at all.

The typical approaches of dealing with missing values depend on the kind of the problem. Thus, we can safely say that there is no good way to deal with missing data. Trying several techniques, building several models and selecting the one that works might be the best.

At the prediction time, if your example is not complete, you should use the same data imputation technique to fill the missing feature as the technique you used to complete the training data.

To sum up, data imputation is tricky and should be done with care. It is important to understand the nature of the data that is missing when deciding which algorithm to use for imputations. 

Based on the origin of the missing data, the following terminology is used to describe it:

* **Missing at Random (MaR)**

  This category of missing data refers to the attributes that could not be answered due to the way the survey was designed. For example, consider the following question in a survey: 
  
  (a) Do you smoke? Yes / No
  (b) If yes, how frequently? Once a week, once a day, twice a day, mote than 2 times in a day. 
  
  You can see that the answer to the question (b) can only be given if the answer to the question (a) is a 'Yes'. This kind of missing values in the dataset arises due to the dependency of one attribute on another attribute.
  
* **Missing Completely at Random (MCaR)**

  This category of missing data is trully missed data or data that was not captured due to the oversight or for other reasons. It means that the missingness is nothing to do with the person being studied. For example, a questionnaire might be lost or a blood sample might be damaged in the lab.
  
* **Missing Not at Random (MNaR)**

  This category of missing data is dependent on the value of the data itself. For example, a survey need people to reveal their 10th grade marks in the chemistry. It may happen that people with lower marks may choose not to reveal them so you would see only high marks in the data sample. 

## Methods

1. Ignore the data row or a whole column
  If you dataset is big enough, you can sacrifice some training examples. This is a quick solution and typically is preferred in cases where the percentage of missing values is relatively low (< 5%). It is a dirty approach as you loose the data.
  
  However dropping one whole observation (row) just because one of the features had a missing value, even if the rest of features are perfectly filled and informative can be a no-no in some cases. Therefore, you can also select to drop the rows only if all the values in the row are missing.
  Sometimes you may want to keep only the rows with at least 4 non-na values (4 is an arbitrary number here).
  Sometimes you may just want to drop a column (variable) that has some missing values. But this approach is valid only if data is missing far more than 60% observations and that variable is insignificant. 

2. Replace with some constant outside the fixed value range, -999, -1 etc...
  Another technique is to replace the missing value with a value outside the normal range of a feature. For example, if the normal range is $[0, 1]$, then you can set the missing value to 2 or -1. The idea is that the learning algorithm will learn what is the best to do when the feature has a value significantly different from the regular values. Alternatively, you can replace the missing value by a value in the middle of the range. For example, if the range for a feature is $[-1, 1]$, you can set the missing value to be equal to 0. Here, the idea is that the value in the middle of the range will not significantly affect the predictions. 
  
3. Replace with mean and median value
  This simple imputation method is based on treating every variable (column) individually, ignoring any inter-relationships with other variables.
  Mean is suitable for continuous data without outliers. Median is suitable for continuous data with outliers.
  For categorical feature, you can select to fill in the missing values with the most common value (mode).
  Note that the mean, median and mode imputation diminishes any correlations involving the variable (s) that are imputed. This is because we assume that there is no relationship between imputed variable and any other measured variables. Thus, those imputations have some attractive properties for univariate analysis but become problematic for multivariate analysis.
  
4. Do nothing
  That is an easy solution. You just let the algorithm handle the missing data. Some algorithms can factor in the missing values and learn the best imputations values for the missing data based on the training loss reduction (ie. XGBoost). Some others have the option to just ignore them (ie. LightGBM — use_missing=false). However, other algorithms will panic and throw an error complaining about the missing values (ie. Scikit learn — LinearRegression). In that case, you will need to handle the missing data and clean it before feeding it to the algorithm.
  
5. Isnull feature

  Adding a new feature `isnull` indicating which rows have missing values for this feature. By doing so, the tree based methods can now understand that there was a missing value. The downside is that we double the number of features.
  
6. Extrapolation and interpolation
  They try to estimate the values from other observations within the range of a discrete set of known data points.
  
7. Linear Regression
  To begin, several predictors of the variable with missing values are identified using a correlation matrix. In other words, we use all other independent variables in the dataset against this variable with missing values. The best predictors are selected and used as independent variables in a regression equation. The variable with missing data is used as dependent variable. Cases with complete data for the predictor variables are used to generate the regression equation. This equation is then used to predict the missing values for incomplete cases. Here, one must assume that there is a linear relationship between the variables used in the regression equation where there may be not be one. 
  
  Similar idea can be applied when we try to impute the categorical variable. Here, we create a predictive model to estimate values that will substitute the missing data. In this case, we divide the dataset into two sets. One set with no missing values (training set) and another one with missing values (testing set). We can then use the classification methods like logistic regression and/or ANOVA for predictions.
  
8. K-nearest Neighbor
  In this method, k neighbors are chosen based on some distance measure and their mode (for categorical variable) /average (for continuous variable) is used as an imputation estimate. It can be used both discrete attributes and continuous attributes. For continuous data, we can use Euclidean Distance, Manhattan Distance and Cosine Similarity metrics. For categorical data, Hamming distance is generally used. For mixed types of data, Gower distance can be selected to calculate the distance. Gower distance uses Manhattan for calculating the distance between continuous data point and Dice distance for calculating the distance between categorical data points. 
  One of the obvious drawbacks of KNN algorithm is that it becomes time consuming when analyzing the larger datasets because it searches for similar instances through the entire set. It is computationally expensive because it stores the whole training dataset in memory. It is also sensitive to outliers. 
  Random Forest algorithm can also be used here. It produces a robust results because it works well with non-linear and categorical data. 
  Stochastic Regression imputation, Hot-deck imputation, Soft-deck imputation, Multivariate Imputation by Chained Equation (MICE) are other approaches to be used. 
  
9. Assigning a unique category
  This approach is valid only for a categorical feature. We assign another class for the missing value. This strategy will add more information into dataset which will result in the change of variance. Since they are categorical, we need to use one-hot encoding to convert it to a numeric form for the algorithm to understand it.
