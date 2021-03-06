# Notes for Machine Learning #

These are my notes for the [Machine Learning course](http://class.coursera.org/ml-003/) given by [Andrew Ng](http://ai.stanford.edu/~ang/) on [Coursera](http://www.coursera.org). For more, please visit http://michen6.github.io/.

## Introduction ##

### What is Machine Learning? ###

> A computer program is said to learn from experience E with respect to some task T and performance measure P, if its performance on T, as measured by P, improves with experience E. - Tom Mitchell

In email spam filtering program, **T** is to classify emails as spams or not spams. **P** is the number of emails correctly classified as spams or not spams. **E** is watching you label emails as spams or not spams.

### Machine Learning Algorithms ###

- **Supervised Learning**: you teach the program
- **Unsupervised Learning**: the program learn by itself
- Reinforcement Learning
- Recommender Systems

Also talk about: Practical advice for applying learning algorithms.

## Classification ##

Example:

- Email: Spam / Not Spam?
- Online Transactions: Fraudulent (Yes / No)?
- Tumor: Malignant / Benign ?

**Binary Classification problem**: \\( y \in \\{0, 1\\} \\) where

- 0: Negative Class (malignant tumor)
- 1: Postive Class (benign tumor)

**Multi-class Classification problem**: \\( y \in \\{0, 1, 2, 3...\\} \\)

- If \\( h\_{\theta}(x) > 0.5 \\), \\( y = 1 \\)
- If \\( h\_{\theta}(x) < 0.5 \\), \\( y = 0 \\)

## Logistic Regression ##

Recall Linear Regression Model funciton: \\( h\_{\theta}(x) = \theta^{T}x \\)

\\( h\_{\theta}(x) = g(\theta^{T}x) \\) where \\( g(z) = \frac{1}{1 + e^{-z}} \\), so
\\( h\_{\theta}(x) = \frac{1}{1 + e^{-\theta^{T}x}} \\).

### Interpretation of Hypothesis Output ###

- \\( h\_{\theta}(x) = \\) estimated probabilities that \\( y = 1 \\) on input \\( x \\).
- \\( h\_{\theta}(x) = P (y = 1 | x; \theta) \\) is interpreted as:
probability that \\( y = 1 \\), given \\( x \\) parameterized by \\( \theta \\)

Since \\( P(y = 0 | x; \theta) + P(y = 1 | x; \theta) = 1 \\), we have \\[ P(y = 0 | x; \theta) = 1 - P(y = 1 | x; \theta) = 1 - h\_{\theta}(x) = 1 - \frac{1}{1 + e^{-\theta^{T}x}} \\]

### Non-linear Decision Boundaries ###

\\[ h\_{\theta}(x) = g(\theta\_{0} + \theta\_{1}x\_{1} + \theta\_{2}x\_{2} + \theta\_{3}x\_{1}^{2} + \theta\_{4}x\_{2}^{2}) \\]

Let \\( \theta\ = \\begin{bmatrix} -1 \\\\ 0 \\\\ 0 \\\\ 1 \\\\ 1 \\end{bmatrix} \\), we have \\( h\_{\theta}(x) = g(-1 + x\_{1}^{2} + x\_{2}^{2}) \\), which predicts \\( y = 1 \\) if \\( -1 + x\_{1}^{2} + x\_{2}^{2} \geq 0 \\).

### Cost Function ###

For Linear Regression, the Cost Function is a squared cost function:
\\[ \text{Cost}(h\_{\theta}(x, y)) = \frac{1}{2}(h\_{\theta}(x) - y)^{2} \\]

For Logistic Regression, the Cost Function is:
\\[ \text{Cost}(h\_{\theta}(x, y)) = \begin{cases} -\log{h\_{\theta}(x)} & \text{if} \; y = 1 \\\\
-\log{(1-h\_{\theta}(x))} & \text{if} \; y = 0 \end{cases} \\]

Recall we have \\( J(\theta) = \frac{1}{m} \sum\_{i=1}^{m} \text{Cost}(h\_{\theta}(x^{(i)}), y^{(i)}) \\) and \\( y \in \\{0, 1\\} \\).

### Gradient Descent ###

Cost Function in one equation:
\\[ \text{Cost}(h\_{\theta}(x, y)) = -y \log{h\_{\theta}(x)} + - (1 - y) \log{(1-h\_{\theta}(x))} \\]

Thus we have
\\[ J(\theta) = \frac{1}{m} \sum\_{i=1}^{m} \text{Cost}(h\_{\theta}(x^{(i)}), y^{(i)}) = - \frac{1}{m} \left[\sum\_{i=1}^{m}y^{(i)} \log{h\_{\theta}(x^{(i)})} + (1 - y^{(i)}) \log{(1-h\_{\theta}(x^{(i)}))} \right] \\]

We will \\( \min_{\theta}J(\theta) \\) and output \\( h\_{\theta}(x) = \frac{1}{1 + e^{-\theta^{T}x}} \\), so we do

Repeat \\( \\{ \\)
\\[ \theta\_{j} := \theta\_{j} - \alpha \frac{\partial}{\partial \theta\_{j}}J(\theta) \qquad (j = 0, 1, 2, 3, ..., n) \\]
\\( \\} \\)
and \\[ \frac{\partial}{\partial \theta\_{j}}J(\theta) = \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_{j}^{(i)} \\] which is identical to the algorithm of linear regression! The difference is the definition of hypothesis \\[ h\_{\theta}(x) = \begin{cases} \theta^{T}x & \text{for linear regression} \\\\ \frac{1}{1 + e^{-\theta^{T}x}} & \text{for logistic regression} \end{cases} \\].

### Advanced Optimization ###

- Gradient descent
- Conjugate gradient
- BFGS
- L-BFGS

Advantages:

-  No need to manually pick \\( \alpha \\). They perform a linear search for all possible \\( \alpha \\)
-  Often faster than gradient descent

Disadvantages:

-  More complex: you should not implement these algorithms by yourself, and if you do, use function to calculate inverse of matrix. Make sure to try different libraries of different implementations

### Multi-class Classification: One-vs-all  ###

Example:

- Email foldering/tagging: Work, Friends, Family, Hobby
- Medical diagrams: Not ill, Cold, flu
- Weather: Sunny, Cloudy, Rain, Snow

We have \\( h\_{\theta}^{(1)}(x) \\) for the first class, \\( h\_{\theta}^{(2)}(x) \\) for the second class and so one for all classes. So we calculate
\\[ h\_{\theta}^{(i)}(x) = P(y=i|x;\theta) \qquad (i = 1, 2, 3) \\]

On a new input \\( x \\), to make a prediction, pick the class \\( i \\) that maximizes the hypothesis \\[ \max\_{i}h\_{\theta}^{(i)}(x) \\]

## Regularization ##

The problem of over-fitting

For linear regression, we can have
\\[ h\_{\theta}(x) = \begin{cases} \theta\_{0} + \theta\_{1}x & \text{underfit} & \text{high bias} \\\\ \theta\_{0} + \theta\_{1}x + \theta\_{2}x^{2} & \text{just right} \\\\ \theta\_{0} + \theta\_{1}x + \theta\_{2}x^{2} + \theta\_{3}x^{3} + \theta\_{4}x^{4} & \text{overfit} & \text{high variance} \end{cases} \\]

- High Bias: 
- High variance: 

### Addressing Overfitting ###

Options:

1. Reduce number of features
	- Manually select which features to keep
	- Model selection algorithm
2. Regularization
	- Keep all the features, but reduce magnitude/values of parameters \\( \theta\_{j} \\)
	- Works well when we have a lot of features, each of which contributes a bit to predicting \\( y \\).

### Cost Function for Regularization ###

Small values for parameters \\( \theta\_{0}, \theta\_{1}, ..., \theta_{n} \\)

- "Simpler" hypothesis
- Less prone to overfitting

We penalize all parameters by adding extra term in cost function:
\\[ J(\theta) = \frac{1}{2m} \left[ \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})^{2} + \lambda \sum\_{j=1}^{n} \theta\_{j}^{2} \right] \\]
in which we usually don't penalize \\( \theta\_{0} \\). \\( \lambda \\) controls the tradeoff between the goal of fitting the training set well and the goal of keeping the parameters small and thus keeping the hypothesis relatively simple to avoid overfitting.

If \\( \lambda \\) is chosen as a very large number, then all the parameters \\( \theta\_{1}, \theta\_{2}, ... \theta\_{n} \\) will be penalized to close to 0, thus the hypothesis is almost equal to \\( h\_{\theta}(x) = \theta\_{0} \\), which will underfit the training set.

### Regularized Linear Regression ###

For Gradient Descent:

Repeat: \\( \\{ \\)
\\[ \begin{aligned} & \theta\_{0} := \theta\_{0} - \alpha \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_{0}^{(i)} \\\\ & \theta\_{j} := \theta\_{j} - \alpha \left[ \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_{j}^{(i)} + \frac{\lambda}{m} \theta\_{j} \right] \\\\ & \qquad\qquad\qquad (j = 1, 2, 3, ..., n) \end{aligned} \\]
\\( \\} \\) in which we can derive:
\\[ \theta\_{j} := \theta\_{j}(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_{j}^{(i)} \\]
where \\( 1 - \alpha \frac{\lambda}{m} \\) is always a number small than 1 which will shrink \\( \theta\_{j} \\) in each iteration.

For Normal Equation:

\\[ \theta = \left(X^{T}X + \lambda \begin{bmatrix}
0 \\\\ 
& 1  \\\\ 
& & \ddots \\\\
& & & 1
\end{bmatrix} \right) ^{-1}  X^{T}y \\]

Note suppose \\( m \leq n \\), \\( X^{T}X \\) is non-invertible/singular, but with regularization, if \\( \lambda > 0 \\), \\( \left(X^{T}X + \lambda \begin{bmatrix}
0 \\\\ 
& 1  \\\\ 
& & \ddots \\\\
& & & 1
\end{bmatrix} \right) \\) is non-singular and thus invertible.

### Regularized Logistic Regression ###

jVal = \\( J(\theta) = \left[- \frac{1}{m} \sum\_{i=1}^{m}y^{(i)} \log{h\_{\theta}(x^{(i)})} + (1 - y^{(i)}) \log{(1-h\_{\theta}(x^{(i)}))} \right] + \frac{\lambda}{2m} \sum\_{j=1}^{n} \theta\_{j}^{2} \\)

gradient(1) = \\( \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)})-y^{(i)})x\_{0}^{(i)} \\)

gradient(2) = \\( \left( \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)})-y^{(i)})x\_{1}^{(i)} \right) + \frac{\lambda}{m} \theta\_{1} \\)

gradient(n+1) = \\( \left( \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)})-y^{(i)})x\_{n}^{(i)} \right) + \frac{\lambda}{m} \theta\_{n} \\)

## Neural Networks ##

Neural networks are the state-of-the-art techniques for many machine learning problems (Non-linear classification).

### Non-linear Hypothesis ###

If we include quadratic in logistic regression, we will have \\( n^{2} \\) features for n original features.

In computer vision, we have a car detection problem. The picture is recorded as a matrix of pixels which is a integer.  For example, if we have a 50x50 pixel images -> 2500 pixels (7500 if RGB).
\\[ x= \begin{bmatrix} \text{pixel 1 intensity}
\\\\ \text{pixel 2 intensity} \\\\ \vdots \\\\ \text{pixel n intensity} \end{bmatrix} \\]
where simple logistic regression will introduce \\( \approx 3 \text{million features} \\).

### Neurons and Brain ###

Neural networks origins as algorithms that try to mimic the brain.

- Auditory Cortex is related to hearing.
- Somatosensory cortex is related to touch.

### Model Representation ###

- Input: Dendrite
- Computation: Nucleus
- Output: Axon
- Transmission: Spites (pulse of electricity)

Neuron model: Logistic Unit

Sigmoid (logistic) activation function
![](img/8-model-representation-1.png)

Neuron network is a group of logistic units.
![](img/8-model-representation-2.png)

\\( a\_{i}^{(j)} = \\) "activation" of unit \\( i \\) in layer \\( j \\).
\\( \Theta^{(j)} = \\) matrix of weights controlling function mapping from layer \\( j \\) to layer \\( j + 1 \\).
![](img/8-model-representation-3.png)

### Forward Propagation: Vectorized Implementation ###

Neural Networks learn its own features.

### Examples ###

- AND

![](img/8-examples-and.png)

- OR

![](img/8-examples-or.png)

- Negation

![](img/8-examples-not.png)

### Multi-class Classification ###

![Multi-class](img/8-multi-class.png)

\\[ y^{(i)} \in \left\\{ \begin{bmatrix} 1 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix} , \begin{bmatrix} 0 \\\\ 1 \\\\ 0 \\\\ 0 \end{bmatrix} , \begin{bmatrix} 0 \\\\ 0 \\\\ 1 \\\\ 0 \end{bmatrix} , \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\\\ 1 \end{bmatrix} \right\\} \\]

 **Q**: Suppose you have a multi-class classification problem with three classes, trained with a 3 layer network. Let \\( a^{(3)}\_1 = (h_\Theta(x))\_1 \\) be the activation of the first output unit, and similarly \\( a^{(3)}\_3 = (h\_\Theta(x))\_3 \\). Then for any input x, it must be the case that \\( a^{(3)}\_1 + a^{(3)}\_2 + a^{(3)}\_3 = 1 \\).

**A**: The outputs of a neural network are not probabilities, so their sum need not be 1.
