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

Naive Bayes classifier works well when variables are "reasonably independent", i.e. they can be a bit correlated but should not be very correlated. For example, the correlation between sepal.width and sepal.length and petal.width and petal.length for the iris dataset is -0.117 versus 0.96 (!). Similar conclusions can be drawn when looking at the diagonal of the covariance matrices for these comparisons. However, for the vowels dataset, columns are formants, and they are of course dependent if measured on the same speaker. 

Naive Bayes classifier is good for moderate or large datasets, for example for medical diagnoses (where symptoms are independent) and classification of text documents or spam-emails (where words are independent).

**(2) How does the decision boundary look for the Iris dataset? How could one improve
the classification results for this scenario by changing classifier or, alternatively,
manipulating the data?**

Classes 0 and 1 are well separated but 1 and 2 are not. Perhaps a SVM with a radial boundary would perform better. One could also try transforming the data to see if the categories become more separable.

<h2>Assignment 4-5</h2>

Implement the Adaboost algorithm and apply it to the Bayes classifier. Design a function that classifiees the instances in data by means of the aggregated boosted classifier according to Equation 15. 

Compute the classification accuracy of the boosted classifier on some data sets and compare it with those of the basic classifier on the vowels and iris data sets. 

**(1) Is there any improvement in classification accuracy? Why/why not?**

**(2) Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?**

**(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?**

<h2>Assignment 6</h2>

Test the decision tree classifier on the vowels and iris data sets. Repeat but now by passing it as an argument to the BoostClassifier object. Answer questions 1-3 in assignment 5 for the decision tree.

**(1) Is there any improvement in classification accuracy? Why/why not?**

**(2) Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?**

**(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?**

<h2>Assignment 7</h2>
If you had to pick a classier, naive Bayes or a decision tree or the boosted versions of these, which one would you pick? Motivate from the following criteria: outliers, irrelevant inputs: part of the feature space is irrelevant, predictive power, mixed types of data: binary, categorical or continuous features, etc., scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both.
