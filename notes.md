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

Let \\( \theta = \\begin{bmatrix} -1 \\\\ 0 \\\\ 0 \\\\ 1 \\\\ 1 \\end{bmatrix} \\), we have \\( h\_{\theta}(x) = g(-1 + x\_{1}^{2} + x\_{2}^{2}) \\), which predicts \\( y = 1 \\) if \\( -1 + x\_{1}^{2} + x\_{2}^{2} \geq 0 \\).

### Cost Function ###

For Linear Regression, the Cost Function is a squared cost function: \\[ Cost(h\_{\theta}(x, y)) = \frac{1}{2}(h\_{\theta}(x) - y)^{2} \\]

For Logistic Regression, the Cost Function is: \\[ f(x) = \\begin{cases} -\\log(h\_{\theta}(x)) & \\text{if} \\; y = 1 \\\\
-\\log(1-h\_{\theta}(x)) & \\text{if} \\; y = 0 \\end{cases} \\]
