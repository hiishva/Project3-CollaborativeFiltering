# Project3-CollaborativeFiltering
## Collaborative Filtering
* Read the attached paper on Empirical Analysis of Predicitive Algorithms for Collaborative Filtering. Read up to section 2.1
* The dataset we will be using is a subset of the movie ratings data from the Netflix Prize. You need to download it via Elearning. It contains a training set, a test set, a movies file, a dataset description file, and a README file. The training and test sets are both subsets of the Netflix training data. You will use the ratings provided in the training set to predict those in the test set. You will compare your predictions with the actual ratings provided in the test set. The evaluation metrics you need to use are the Mean Absolute Error and the Root Mean Squared Error. The dataset description file further describes the dataset, and will help you get started. The README file is from the original set of Netflix files, and has been included to comply with the terms of use for this data.
* Implement (use Python3 and numpy; the latter is a must for this part) the collaborative filtering algorithm described in Section 2.1 of the paper (Equations 1 and 2; ignore Section 2.1.2) for making the predictions.

## K-Nearest Neighbor and SVMs
* For this part, you will use scikit learn.
* Download the MNIST dataset via scikit-learn, see below on how to do it (you did this for project 2 as well). The dataset has a training set of 60,000 examples, and a test set of 10,000 examples where the digits have been centered inside 28x28 pixel images. You can also use scikit-learn to download and rescale the dataset using the following code:
```
from sklearn.datasets import fetch_openml
#Load data from the website
X, y = fetch_openml(’mnist_784’, version=1, return_X_y=True)
X = X / 255.
# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```
* Use the SVM classifier in scikit learn and try different kernels and values of penalty parameter. Important: Depending on your computer hardware, you may have to carefully select the parameters (see the documentation on scikit learn for details) in order to speed up the computation. Report the error rate for at least 10 parameter settings that you tried. Make sure to precisely describe the parameters used so that your results are reproducible.
* Use the k Nearest Neighbors classifier called KNeighborsClassifier in scikit learn and try different parameters (see the documentation for details). Again depending on your computer hardware, you may have to carefully select the parameters in order to speed up the computation. Report the error rate for at least 10 parameters that you tried. Make sure to precisely describe the parameters used so that your results are reproducible.
* What is the best error rate you were able to reach for each of the three classifiers? Note that many parameters do not affect the error rate and we will deduct points if you try them. It is your duty to read the documentation and then employ your machine learning knowledge to determine whether a particular parameter will affect the error rate. Finally, don’t change just one parameter 10 times; we want to see diversity.
