# Notes for Machine Learning #

This is my notes for the [Machine Learning course](http://class.coursera.org/ml-003/) given by [Andrew Ng](http://ai.stanford.edu/~ang/) on [Coursera](http://www.coursera.org).

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

Recall we have \\( J(\theta) = \frac{1}{m} \sum\_{i=1}^{m} \text{Cost}(h\_{\theta}(x^{(i)}), y^{(i)}) \\) and \\( y = 0 \\) or \\( 1 \\) always.

### Gradient Descent ###

Cost Function in one equation:
\\[ \text{Cost}(h\_{\theta}(x, y)) = -y \log{h\_{\theta}(x)} + - (1 - y) \log{(1-h\_{\theta}(x))} \\]

Thus we have
\\[ J(\theta) = \frac{1}{m} \sum\_{i=1}^{m} \text{Cost}(h\_{\theta}(x^{(i)}), y^{(i)}) = - \frac{1}{m} \left[\sum\_{i=1}^{m}y^{(i)} \log{h\_{\theta}(x^{(i)})} + (1 - y^{(i)}) \log{(1-h\_{\theta}(x^{(i)}))} \right] \\]

We will \\( \min_{\theta}J(\theta) \\) and output \\( h\_{\theta}(x) = \frac{1}{1 + e^{-\theta^{T}x}} \\), so we do

\\[ \text{Repeat} \\{ \theta\_{j} := \theta\_{j} - \alpha \frac{\partial}{\partial \theta\_{j}}J(\theta) \\} \\]
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
\\[ h\_{\theta}^{(i)}(x) = P(y=i|x;\theta) \hspace{20} (i = 1, 2, 3) \\]

On a new input \\( x \\), to make a prediction, pick the class \\( i \\) that maximizes \\[ \max\_{i}h\_{\theta}^{(i)}(x) \\]

## Regularization ##

The problem of over-fitting

For linear regression, we can have
\\[ h\_{\theta}(x) = \begin{cases} \theta\_{0} + \theta\_{1}x & \text{linear is underfit and high bias} \\\\ \theta\_{0} + \theta\_{1}x + \theta\_{2}x^{2} & \text{quadratic is just right} \\\\ \theta\_{0} + \theta\_{1}x + \theta\_{2}x^{2} + \theta\_{3}x^{3} + \theta\_{4}x^{4} & \text{overfit and high variance} \end{cases} \\]

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
\\[ J(\theta) = \frac{1}{2m} \left[ \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})^{2} + \lambda \sum\_{j=1}^{n} \theta\_{j}^{2} \right] \\] in which we usually don't penalize \\( \theta\_{0} \\). \\( \lambda \\) controls the tradeoff between the goal of fitting the training set well and the goal of keeping the parameters small and thus keeping the hypothesis relatively simple to avoid overfitting.

