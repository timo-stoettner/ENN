# ENN
Python Implementation of the Extended Nearest Neighbor Algorithm (ENN) proposed by Tang and He (2015), which constitutes an improvement to the widespread KNN algorithm. It has been shown that ENN can lead to significantly more accurate results than KNN and other popular algorithms for different kinds of datasets. For more information, see the paper at  http://www.ele.uri.edu/faculty/he/PDFfiles/ENN.pdf

# Usage

To use the classifier, simply import the ENN class by typing

```python
from ENN import ENN
```

The classifier can be used just like classifiers implemented in the scikit-learn package and therefore contains a fit and a predict method, among others. To obtain the predictions of the classifier, simply type

```python
clf = ENN()
clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)
```

The constructor takes two parameters:

- `k`: The number of nearest neighbors to consider (*standard: 3*)
- `distance_function`: The distance function used to compute the distances between records. It needs to take two 1-d arrays as arguments and return a scalar value indicating the distance between the arrays. The standard function is the euclidean distance function implemented in scipy.spatial.distance. 

Therefore, the following code is also valid:

```python
from scipy.spatial.distance import mahalanobis
clf = ENN(k=7, distance_function=mahalanobis)
clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)
```

Since the classifier is very similar to sklearn classifiers, it can also be used in connection with other sklearn methods such as Pipelines or Grid Searches.

```python
from sklearn.grid_search import GridSearchCV
from scipy.spatial.distance import euclidean, mahalanobis
clf = GridSearchCV(ENN(), {'k' : [3,5,7,8], "distance_function": [euclidean, hamming]}) 
clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)
```

# Performance

The performance of the distance function used is essential to the overall performance of the classifier. Therefore, when performance is an issue, I recommend making use of an optimized distance function. The easiest and at the same time most performant way to optimize the distance function I've come across so far is by making use of the package Numba.

For instance, for euclidean distances you can make use of the following code snippet...

```python
from numba.decorators import autojit
def euclidean(x,y):   
    return np.sqrt(np.sum((x-y)**2))
optimized_euclidean = autojit(euclidean)
```

... and then simply pass the optimized distance function to the classifier.

```python
clf = ENN(distance_function = optimized_euclidean)
```

This is significantly faster than the scipy distance function used as the standard function of the classifier. However, Numba can be challenging to compile when not installed together with a Python distribution such as Anaconda and therefore was not used as the standard function for the distance computation. 


# Compatibility

Tested with Python 2.7 and sklearn 0.17.1. If you run into any issues, feel free to let me know.
