+++
date = "2017-06-07T21:00:11+07:00"
title = "Linear Regression for Machine Learning"
type = "post"
categories = ["travel", "photography"]

+++


This is the first in a series of posts in which I'll be demonstrating concepts from Andrew Ng's [Introduction to Machine Learning](https://www.coursera.org/learn/machine-learning) course on Coursera.  As a CS student, I enjoyed doing odd or extra things with my assignments, I found it added a greater challenge and allowed me to learn things outside the scope of the class in a well-structured way.  So, as I progress through the course, I'll be posting an extended analogue of each of the assignments, translated into Python in a Jupyter notebook.

Each post will begin with an implementation of the algorithm for the week, tracking closely to the assignment but trading Octave/MATLAB functions for standard Python data science tools, and then conclude by exploring what the same algorithm would look like built in [Tensorflow](https://www.tensorflow.org/).  I'll also be using different data sets, to make things a bit more interesting.

The first assignment covers [linear regression](https://en.wikipedia.org/wiki/Linear_regression).  Linear regression attempts to fit a line of best fit to a data set, using one or more features as coefficients for a linear equation.  Specifically, I'll discuss:

 - Loading, manipulating and plotting data using numpy and matplotlib
 - The cost function and gradient descent for simple linear regression with one variable
 - Feature scaling
 - Gradient descent for multiple variables, vectorization and the normal equation
 - Representing linear regression and gradient descent in Tensorflow

The data set this week will be the [UCI Bike Sharing Data Set](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

# Loading and Plotting Data

For the first part, we'll use only two fields from the daily data set: `atemp`, the normalized temperature in C, and `cnt`, the total number of bike rentals.  The rentals are scaled by a thousand given the large difference in magnitude between them and the normalized temperatures:


```python
import pandas as pd

data = pd.read_csv("./data.csv")
temps = data['atemp'].values
rentals = data['cnt'].values / 1000
```

The plot reveals some obvious, though loose correlation between temperature and bike rentals, as would be expected:


```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(temps, rentals, marker='x', color='red')
plt.xlabel('Normalized Temperature in C')
plt.ylabel('Bike Rentals in 1000s')
```




    <matplotlib.text.Text at 0x118bda3c8>




![png](/images/ml-linear-regression/ml-linear-regression_5_1.png)


# Simple Linear Regression

We'll start by implementing the [cost function](https://en.wikipedia.org/wiki/Loss_function) for linear regression, specifically [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE).  Intuitively, MSE represents the aggregated distances of each point's actual y-value from what a hypothesis function $h_\theta(x)$ predicted it would be.  That hypothesis function and the cost function $J(\theta)$ are defined as

`\begin{align}
h_\theta(x) & = \theta^Tx = \theta_0 + \theta_1x_1 \\
J(\theta) & = \frac{1}{2m}\sum\limits_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
\end{align}`

where $\theta$ is a vector of features, and $x^{(i)}$ is the ith training example, $x_j$ is its value for the jth feature, and $y^{(i)}$ is the actual y-value for the example.


```python
import numpy as np

def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))
```

Before computing the cost with an initial guess for $\theta$, a column of 1s is prepended into the input data.  This allows us to vectorize the function, as well as make it usable for multiple linear regression later.  This first value $\theta_0$ becomes a constant in the cost function.


```python
theta = np.zeros(2)
X = np.column_stack((np.ones(len(temps)), temps))
y = rentals
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)
```

    theta: [ 0.  0.]
    cost: 12.0184064412


We'll minimize the cost using the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm.  Intuitively, gradient descent takes small, linear hops down the slope of a function in each dimension, coverging on a global minimum.  This is accomplished by updating the feature vector using the partial derivative of the cost function with respect to each feature, as shown here for two feature dimensions:

`\begin{align}
\theta_0 & := \theta_0 - \alpha\frac{\partial}{\partial\theta_0} J(\theta_0,\theta_1) \\
\theta_1 & := \theta_1 - \alpha\frac{\partial}{\partial\theta_1} J(\theta_0,\theta_1)
\end{align}`

with $\alpha$ specifying the *learning rate*, here adjusting the size of the hops the algorithm makes.  The update rule then becomes:

`\begin{align}
\theta_0 & := \theta_0 - \alpha\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)}) \\
\theta_1 & := \theta_1 - \alpha\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_1^{(i)} \\
\end{align}`

See [here](http://mccormickml.com/2014/03/04/gradient-descent-derivation/) for a more detailed explanation of the math.


```python
def gradient_descent(X, y, alpha, iterations):
    theta = np.zeros(2)
    m = len(y)

    for i in range(iterations):
        t0 = theta[0] - (alpha / m) * np.sum(np.dot(X, theta) - y)
        t1 = theta[1] - (alpha / m) * np.sum((np.dot(X, theta) - y) * X[:,1])
        theta = np.array([t0, t1])

    return theta

iterations = 5000
alpha = 0.1

theta = gradient_descent(X, y, alpha, iterations)
cost = compute_cost(X, y, theta)

print("theta:", theta)
print('cost:', compute_cost(X, y, theta))
```

    theta: [ 0.94588081  7.50171673]
    cost: 1.12758692584


Now, let's try a few different visualizations of the resulting values for $\theta$, first by plotting $h_\theta(x)$ against the input data:


```python
plt.scatter(temps, rentals, marker='x', color='red')
plt.xlabel('Normalized Temperature in C')
plt.ylabel('Bike Rentals in 1000s')
samples = np.linspace(min(temps), max(temps))
plt.plot(samples, theta[0] + theta[1] * samples)
```




    [<matplotlib.lines.Line2D at 0x118cf41d0>]




![png](/images/ml-linear-regression/ml-linear-regression_14_1.png)


Looks about as expected.

A surface plot is a better illustration of how gradient descent approaches a global minimum, plotting the values for $\theta$ against their cost.  This requires a bit more code than an implementation in Octave / MATLAB, largely in how the input data is generated:


```python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

Xs, Ys = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-40, 40, 50))
Zs = np.array([compute_cost(X, y, [t0, t1]) for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
Zs = np.reshape(Zs, Xs.shape)

fig = plt.figure(figsize=(7,7))
ax = fig.gca(projection="3d")
ax.set_xlabel(r't0')
ax.set_ylabel(r't1')
ax.set_zlabel(r'cost')
ax.view_init(elev=25, azim=40)
ax.plot_surface(Xs, Ys, Zs, cmap=cm.rainbow)
```




    <mpl_toolkits.mplot3d.art3d.Poly3DCollection at 0x118de6a20>




![png](/images/ml-linear-regression/ml-linear-regression_16_1.png)


Lastly, a countour plot stacks slices of that function in 2D space, and can show the resulting $\theta$ values sitting exactly at the minimum:


```python
ax = plt.figure().gca()
ax.plot(theta[0], theta[1], 'r*')
plt.contour(Xs, Ys, Zs, np.logspace(-3, 3, 15))
```




    <matplotlib.contour.QuadContourSet at 0x1192909b0>




![png](/images/ml-linear-regression/ml-linear-regression_18_1.png)


# Multiple Linear Regression

We'll now add two more input features, humidity and windspeed.

Before implementing gradient descent for multiple variables, we'll also apply [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) to normalize feature values, preventing any one of them from disproportionately influencing the results, as well as helping gradient descent converge more quickly.  In this case, each feature value adjusted by subtracting the mean and dividing the results by the standard deviation of all values for that feature:

`$$
z = \frac{x - \mu}{\sigma}
$$``

More detail on feature scaling and normalization can be found [here](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html).


```python
def feature_normalize(X):
    n_features = X.shape[1]
    means = np.array([np.mean(X[:,i]) for i in range(n_features)])
    stddevs = np.array([np.std(X[:,i]) for i in range(n_features)])
    normalized = (X - means) / stddevs

    return normalized

X = data.as_matrix(columns=['atemp', 'hum', 'windspeed'])
X = feature_normalize(X)
X = np.column_stack((np.ones(len(X)), X))

y = data['cnt'].values / 1000
```

The next step is to implement gradient descent for any number of features.  Fortunately, the update step generalizes easily, becoming:

`$$
\theta_j := \theta_j - \alpha\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$`



```python
def gradient_descent_multi(X, y, theta, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)

    for i in range(iterations):
        cost = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * cost

    return theta

theta = gradient_descent_multi(X, y, theta, alpha, iterations)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost', cost)
```

    theta: [ 4.50434884  1.22203893 -0.45083331 -0.34166068]
    cost 1.00587092471


Although we can't evaluate the results visually, we can check them against another implementation of the algorithm, the [normal equation](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/).  This solves directly for the solution without iteration, and without needing to specify an $\alpha$ value, although it begins to perform worse than gradient descent with large (10,000+) numbers of features:

$$
\theta = (X^TX)^{-1}X^Ty
$$


```python
from numpy.linalg import inv

def gradient_descent_multi_normal(X, y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

theta = gradient_descent_multi_normal(X, y)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)
```

    theta: [ 4.50434884  1.22203893 -0.45083331 -0.34166068]
    cost: 1.00587092471


The $\theta$ values and costs for each implementation identical, so we can be confident the implementations are correct.

## Linear Regression in Tensorflow

Tensorflow represents this algorithm as a computational graph, and is capable of minimizing the cost with a built-in optimizer.

We'll begin here by reloading the data and adapting it to more Tensorflow-friendly data structures and terminology.  Features are still normalized as before, but the added column of 1s is absent: the constant is treated separately as a *bias* variable, the previous $\theta$ values are now *weights*:


```python
import tensorflow as tf

X = data.as_matrix(columns=['atemp', 'hum', 'windspeed'])
X = feature_normalize(X)
y = data['cnt'].values / 1000
y = y.reshape((-1, 1))

m = X.shape[0]
n = X.shape[1]

examples = tf.placeholder(tf.float32, [m,n])
labels = tf.placeholder(tf.float32, [m,1])
weights = tf.Variable(tf.zeros([n,1], dtype=np.float32), name='weight')
bias = tf.Variable(tf.zeros([1], dtype=np.float32), name='bias')
```

The gradient descent occurs below in only three lines of code, with the optimizer able to figure out how to minimize the cost function itself:


```python
hypothesis = tf.add(tf.matmul(examples, weights), bias)
cost = tf.reduce_sum(tf.square(hypothesis - y)) / (2 * m)
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
```

All that remains is to run the optimizer:


```python
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1, iterations):
        sess.run(optimizer, feed_dict={
            examples: X,
            labels: y
        })

    print('bias:', sess.run(bias))
    print('weights:', sess.run(weights))
```

    bias: [ 4.50434685]
    weights: [[ 1.22203839]
     [-0.45083305]
     [-0.34166056]]


The bias and weight values correspond to what we got for $\theta$ in both implementations previously, so we know the Tensorflow implementation is working as well.