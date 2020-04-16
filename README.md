<h1>DD2421 LAB 3 - NAIVE BAYES CLASSIFIER + BOOSTING </h1>

<h2> Assignment 1 </h2>
Use the provided function, genBlobs(), that returns Gaussian distributed data points together with class labels, to generate some test data. Compute the ML-estimates for the data and plot the 95%-confidence interval using the function plotGaussians.

![ass1](https://user-images.githubusercontent.com/1690217/78283799-a2dbbd80-751e-11ea-84b8-959f619d1187.png)

This is an example of a Naive Bayes classifier using the Maximum Likelihood Method on Gaussian test data. 

<h2> Assignment 2-3 </h2>
Run testClassifier for the datasets and take note of the accuracies. Use plotBoundary
to plot the decision boundary of the 2D iris dataset. 

```
testClassifier(BayesClassifier(), dataset='iris', split=0.7)

Trial: 0 Accuracy 84.4
Trial: 10 Accuracy 95.6
Trial: 20 Accuracy 93.3
Trial: 30 Accuracy 86.7
Trial: 40 Accuracy 88.9
Trial: 50 Accuracy 91.1
Trial: 60 Accuracy 86.7
Trial: 70 Accuracy 91.1
Trial: 80 Accuracy 86.7
Trial: 90 Accuracy 91.1
Final mean classification accuracy  89 with standard deviation 4.16
```
![ass2](https://user-images.githubusercontent.com/1690217/78284002-f948fc00-751e-11ea-8f07-01984d4c66a6.png)


```
testClassifier(BayesClassifier(), dataset='vowel', split=0.7)

Trial: 0 Accuracy 61
Trial: 10 Accuracy 66.2
Trial: 20 Accuracy 74
Trial: 30 Accuracy 66.9
Trial: 40 Accuracy 59.7
Trial: 50 Accuracy 64.3
Trial: 60 Accuracy 66.9
Trial: 70 Accuracy 63.6
Trial: 80 Accuracy 62.3
Trial: 90 Accuracy 70.8
Final mean classification accuracy  64.7 with standard deviation 4.03
```
![ass3](https://user-images.githubusercontent.com/1690217/78285843-01a23680-7521-11ea-8cfb-fd278f856866.png)

**(1) When can a feature independence assumption be reasonable and when not?**

Naive Bayes classifier is good for moderate or large datasets, for example for medical diagnoses (where symptoms are independent) and classification of text documents or spam-emails (where words are independent). 

Naive Bayes classifier works well when variables are "reasonably independent", i.e. they can be a bit correlated but should not be very correlated. It also assumes that the covariance matrix is diagonal.

Naive Bayes does not work well when there is no occurence between a certain class label and a feature, which leads to a likelihood equal to 0 (e.g. if class that comes in testing data has not been seen in training, we will have zero probability of that particular class). 

**(2) How does the decision boundary look for the Iris dataset? How could one improve
the classification results for this scenario by changing classifier or, alternatively,
manipulating the data?**

Classes 0 and 1 are well separated but 1 and 2 are not. Perhaps a SVM with a radial kernel would perform better. One could also try transforming the data to see if the categories become more separable. Or, one could use boosting, which implements weights that could adjust the importance of the points that are used to define the decision boundary.

<h2>Assignment 4-5</h2>

Implement the Adaboost algorithm and apply it to the Bayes classifier. Design a function that classifies the instances in data by means of the aggregated boosted classifier according to Equation 15. 

Compute the classification accuracy of the boosted classifier on some data sets and compare it with those of the basic classifier on the vowels and iris data sets. 

```
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)

Trial: 0 Accuracy 95.6
Trial: 10 Accuracy 100
Trial: 20 Accuracy 93.3
Trial: 30 Accuracy 91.1
Trial: 40 Accuracy 97.8
Trial: 50 Accuracy 93.3
Trial: 60 Accuracy 93.3
Trial: 70 Accuracy 97.8
Trial: 80 Accuracy 95.6
Trial: 90 Accuracy 93.3
Final mean classification accuracy  94.7 with standard deviation 2.82
```

```
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)

Trial: 0 Accuracy 76.6
Trial: 10 Accuracy 86.4
Trial: 20 Accuracy 83.1
Trial: 30 Accuracy 80.5
Trial: 40 Accuracy 72.7
Trial: 50 Accuracy 76
Trial: 60 Accuracy 81.8
Trial: 70 Accuracy 82.5
Trial: 80 Accuracy 79.9
Trial: 90 Accuracy 83.1
Final mean classification accuracy  80.2 with standard deviation 3.52
```

**(1) Is there any improvement in classification accuracy? Why/why not?**
Yes the classification is improved for the both of the data sets. This is because the Adaboost algorithm takes weights into account, i.e. how important each classifier should be. Aggregated weak classifiers can thus perform as a strong classifier.

* IRIS 
  * NAIVE BAYES: Final mean classification accuracy  89 with standard deviation 4.16
  * NAIVE BAYES + ADABOOST: Final mean classification accuracy  94.7 with standard deviation 2.82

* VOWEL 
  * NAIVE BAYES: Final mean classification accuracy  64.7 with standard deviation 4.03
  * NAIVE BAYES + ADABOOST: Final mean classification accuracy  80.2 with standard deviation 3.52

Boosting involves incrementally building an ensemble by training each new model instance to emphasize the training instancees that previous models mis-classified. This means that we subsequently give more and more weight to observations that are hard to classify. Combining predictors of several estimators generally gives better results than using a single estimator. 

**(2) Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?**

![ass2](https://user-images.githubusercontent.com/1690217/78284002-f948fc00-751e-11ea-8f07-01984d4c66a6.png)
![ass5](https://user-images.githubusercontent.com/1690217/78816303-d496cd80-79d1-11ea-9ab3-eefbcdc14b48.png)

The decision boundary looks a bit more complex. In particular, it separates a certain area where there were errors in classification without boosting erroneously confused class 1 and 2. The boosting allows the model to focus more on these previously misclassified points.

**(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?**

Yes. Boosting can be used to reduce bias and variance, but it can also result in overfitting. The idea is to turn a weak classifier into a strong one. 

<h2>Assignment 6</h2>

Test the decision tree classifier on the vowels and iris data sets. Repeat but now by passing it as an argument to the BoostClassifier object. Answer questions 1-3 in assignment 5 for the decision tree. 

```
testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)

Trial: 0 Accuracy 95.6
Trial: 10 Accuracy 100
Trial: 20 Accuracy 91.1
Trial: 30 Accuracy 91.1
Trial: 40 Accuracy 93.3
Trial: 50 Accuracy 91.1
Trial: 60 Accuracy 88.9
Trial: 70 Accuracy 88.9
Trial: 80 Accuracy 93.3
Trial: 90 Accuracy 88.9
Final mean classification accuracy  92.4 with standard deviation 3.71
```

```
testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)

Trial: 0 Accuracy 95.6
Trial: 10 Accuracy 100
Trial: 20 Accuracy 95.6
Trial: 30 Accuracy 93.3
Trial: 40 Accuracy 93.3
Trial: 50 Accuracy 95.6
Trial: 60 Accuracy 88.9
Trial: 70 Accuracy 93.3
Trial: 80 Accuracy 93.3
Trial: 90 Accuracy 93.3
Final mean classification accuracy  94.6 with standard deviation 3.65
```


```
testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)

Trial: 0 Accuracy 63.6
Trial: 10 Accuracy 68.8
Trial: 20 Accuracy 63.6
Trial: 30 Accuracy 66.9
Trial: 40 Accuracy 59.7
Trial: 50 Accuracy 63
Trial: 60 Accuracy 59.7
Trial: 70 Accuracy 68.8
Trial: 80 Accuracy 59.7
Trial: 90 Accuracy 68.2
Final mean classification accuracy  64.1 with standard deviation 4
```

```
testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)

Trial: 0 Accuracy 84.4
Trial: 10 Accuracy 89.6
Trial: 20 Accuracy 86.4
Trial: 30 Accuracy 93.5
Trial: 40 Accuracy 84.4
Trial: 50 Accuracy 79.9
Trial: 60 Accuracy 89
Trial: 70 Accuracy 86.4
Trial: 80 Accuracy 85.7
Trial: 90 Accuracy 85.7
Final mean classification accuracy  86.6 with standard deviation 2.97
```

**(1) Is there any improvement in classification accuracy? Why/why not?**

Yes there is an improvement for both datasets. However, the improvement is larger for the vowel dataset. 

**(2) Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?**

![ass6a](https://user-images.githubusercontent.com/1690217/78816872-9b129200-79d2-11ea-8aef-12d3dd019338.png)
![ass6b](https://user-images.githubusercontent.com/1690217/78816876-9bab2880-79d2-11ea-9e28-76869483ded5.png)

It actually looks a bit more complex, yes. 

**(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?**

See answer for Assignment 5. 

<h2>Assignment 7</h2>
If you had to pick a classifier, Naive Bayes (NB) or a Decision Tree (DT) or the boosted versions of these, which one would you pick? 

Motivate from the following criteria: 

* ***Outliers***: DTs. There is a risk that boosting will give a lot of importance to the outliers through high weights: outliers can be bad for boosting because boosting builds each new weighted model on previous errors (outliers will have larger residuals than non-outliers, so gradient boosting will focus a disproportionate amount of its attention on those points). NB is sensitive to outliers. DTs are rather robust to outliers (they are based on splitting points in the data, not extreme values). 
* ***Irrelevant inputs** (part of the feature space is irrelevant): For NB, you pick features that matter. For DTs, they tend to ignore irrelevant features. NV outperform decision trees when it comes to rare occurences (e.g. imagine predicting cancer in the general population, a DC will probably prune such important classes out of the model). AdaBoost training process selects only those features known to improve the predictive power of the model, reducing dimensionality and potentially improving execution time as irrelevant features do not need to be computed.
* ***Predictive power***: NB has high predictive power considering how simple it is to more complex models. Combine with boosting to get even better results. DTs can be pruned to improve predictive power. Boosting in general is a technique that is used to create stronger models.
* ***Mixed types of data*** (binary, categorical or continuous features): DTs are known to handle a mix of different features. NB can also handle this, assuming Gaussian and categorical error distributions, but they are more known for continuous data. Boosting can probably improve performance. 
* ***Scalability*** (the dimension of the data, D, is large or the number of instances, N, is large, or both): If your data set is small, NB can achieve reasonable performance. It works well for large data sets too. DT works better for lots of data (small variations in the data can result in a completely different tree since DTs have high variance). NB is not affected by large feature sets. DTs may be hard to interpret if they have a lot of features: more branches on a tree lead to a risk of over-fitting. DTs work best for small number of classes.

In general, Decision Trees have high variance and low bias. Naive Bayes, on the other hand, have high bias and low variance. Naive Bayes will only work if the decision boundary is linear, elliptic, or parabolic. Alternative methods are KNNs. 

