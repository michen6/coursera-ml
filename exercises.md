# Exercise for Machine Learning #

These my solutions to exercises of [Machine Learning course](http://class.coursera.org/ml-003/) given by [Andrew Ng](http://ai.stanford.edu/~ang/) on [Coursera](http://www.coursera.org). For more, please visit http://michen6.github.io/.

# Programming Exercise 1: Linear Regression #

## 1. Simple octave function ##

### Return 5x5 identity matrix in `warmUpExercises.m` ###

  A = eye(5);

## 2. Linear regression with one variable ##

### Loading the Data ###

	data = load('ex1data1.txt'); 		% read comma separated data
	X = data(:, 1); y = data(:, 2);
	m = length(y);						% number of training examples

### Plotting the Data in `plotData.m` ###

	plot(x, y, 'rx', 'MarkerSize', 10); 		% Plot the data
	ylabel('Profit in $10,000s'); 				% Set the y axis label
	xlabel('Population of City in 10,000s'); 	% Set the x axis label

### Gradient Descent ###

Cost Funtion: \\( J(\theta) = \frac{1}{2m} \sum\_{i=1}^{m} (h\_{\theta}(x^{(i)}) - y^{(i)})^{2} \\) where the hypothesis \\( h\_{\theta}(x^{(i)}) \\) is given by \\( h\_{\theta}(x^{(i)}) = \theta^{T}x = \theta\_{0} + \theta\_{1}x\_{1} \\).

In batch gradient descent, each iteration performs the update:
\\[ \theta\_{j} := \theta\_{j} - \alpha \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_{j}^{(i)} \qquad \text{(simultaneously update } \theta\_{j} \text{ for all j )} \\]

**Implementation**

	X = [ones(m, 1), data(:,1)]; 	% Add a column of ones to x
	theta = zeros(2, 1); 			% initialize fitting parameters
	iterations = 1500;
	alpha = 0.01;

### Compute the cost \\( J(\theta) \\) in `computeCost.m` ###

My solution uses `sum` which sum up each column and `.^` which is power by element.:

	J = sum((X * theta - y) .^ 2) / (2 * size(X, 1));	% Compute cost for X and y with theta

This solution creates local variables for hypothesis and cost function:

	h = X*theta;			% Define hypothesis
	c = (h-y).^2;			% Define cost function
	J = sum(c)/(2*m);

or this uses product of matrix to get power and sum:

	J = (1/(2*m)) * (X*theta-y)' * (X*theta-y);

### Gradient Descent in `gradientDecent.m`###

My solution uses two transpose `'` to perform matrix product:

	for iter = 1:num_iters
		theta = theta - alpha / m * ((X * theta - y)' * X)';	% Update theta by gradient descent
	    J_history(iter) = computeCost(X, y, theta);			    % Save the cost J in every iteration
	end

### Visualizing \\( J(\theta) \\) ###

	% initialize J vals to a matrix of 0's
	J vals = zeros(length(theta0 vals), length(theta1 vals));
	% Fill out J vals
	for i = 1:length(theta0 vals)
		for j = 1:length(theta1 vals)
			t = [theta0 vals(i); theta1 vals(j)];
			J vals(i,j) = computeCost(x, y, t);
		end
	end

## Linear regression with multiple variables ##

### Feature Normalization in `featureNormalize.m` ###

Two tasks:

- Subtract the mean value of each feature from the dataset.
- After subtracting the mean, additionally scale (divide) the feature values
by their respective "standard deviations". In Octave, you can use the `std` function to
compute the standard deviation.

My solution uses `repmat` to duplicate `mu` and `sigma` to fit the size of `X`:

	mu = mean(X);
	sigma = std(X);
	mu_tiled = repmat(mu, [size(X, 1), 1]);
	sigma_tiled = repmat(sigma, [size(X, 1), 1]);
	X_norm = (X - mu_tiled)./sigma_tiled

I haven't do the rest of extra parts of ex1:

- `computeCostMulti.m`
- `gradientDescentMulti.m`
- `normalEqn.m`

# Programming Exercise 2: Logistic Regression #

## Logistic Regression ##

### Plotting data in `plotData.m` ###

	% Find Indices of Positive and Negative Examples
	pos = find(y==1); neg = find(y == 0);

	% Plot Examples
	plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ... 'MarkerSize', 7);
	plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ... 'MarkerSize', 7);

**Implementation**

### Sigmoid function in `sigmoid.m` ###

The sigmoid function is defined as \\( g(z) = \frac{1}{1 + e^{-z}} \\)

My solution is using nested loop to compute by each element:

	for i = 1 : size(z, 1)
		for  j = 1 : size(z, 2)
			g(i, j) = 1 / (1 + e ^ ( 0 - z(i, j)));
		end
	end

This solution is straightforward which uses `exp` to calculate the e to the power of -z:

	g = 1.0 ./ (1.0 + exp(-z));

### Cost function and gradient in `costFunction.m` ###

Recall that the cost function in logistic regression is
\\[ J(\theta) = \frac{1}{m} \sum\_{i=1}^{m} \text{Cost}(h\_{\theta}(x^{(i)}), y^{(i)}) = - \frac{1}{m} \left[\sum\_{i=1}^{m}y^{(i)} \log{h\_{\theta}(x^{(i)})} + (1 - y^{(i)}) \log{(1-h\_{\theta}(x^{(i)}))} \right] \\]

My solution is to first calculate hypothesis using `sigmoid` and then apply matrix product to get \\( J(\theta) \\) and gradient vector:

	h = sigmoid(X * theta);									% Define hypothesis
	J = (1/m) * (-y' * log(h) - (1 - y') * log(1 - h))
	grad = (1/m) * (X' * (h - y));

This solution resembles my solution by defining local variables for `costPos` which is 0 when \\( y = 0 \\) and `costNeg` which is 1 when \\( y = 1 \\):

	h = sigmoid(X * theta);		% get the hypothesis for all of X, given theta;
	costPos = -y' * log(h);
	costNeg = (1 - y') * log(1 - h);
	J = (1/m) * (costPos - costNeg);

This solution uses `sum` and `.*` to calculate the sum:

	H_theta = sigmoid(X * theta);
	J = (1.0/m) * sum(-y .* log(H_theta) - (1.0 - y) .* log(1.0 - H_theta));
	grad = (1.0/m) .* X' * (H_theta - y);

### Learning parameters using `fminunc` ###

Octave’s `fminunc` is an optimization solver that finds the minimum of an unconstrained function. For logistic regression, you want to optimize the cost function \\( J(\theta) \\) with parameters \\( \theta \\).

In `ex2.m`, the code is written with the correct arguments:

	% Set options for fminunc
	options = optimset('GradObj', 'on', 'MaxIter', 400);
	
	% Run fminunc to obtain the optimal theta
	% This function will return theta and the cost
	[theta, cost] = ...
		fminunc(@(t)(costFunction(t, X, y)), initial theta, options);

### Evaluating logistic regression in `predict.m` ###

My solution is simply using `round`:

	p = round(sigmoid(X * theta));

## Regularized logistic regression ##

### Feature mapping in `mapFeature.m` ###

Provided function `mapFeature.m` maps the features into
all polynomial terms of \\( x\_{1} \\) and \\( x\_{2} \\) up to the sixth power:

	function out = mapFeature(X1, X2)
		degree = 6;
		out = ones(size(X1(:,1)));
		for i = 1:degree
		    for j = 0:i
		        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
		    end
		end
	end

### Cost function and gradient in `costFunctionReg.m` ###

Recall that the regularized cost function in logistic regression is
\\[ J(\theta) = \frac{1}{2m} \left[ \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)}) - y^{(i)})^{2} + \lambda \sum\_{j=1}^{n} \theta\_{j}^{2} \right] \\]

The gradient of
the cost function is a vector where the \\( j^{th} \\) element is defined as follows:
\\[ \frac{\partial J(\theta)}{\partial \theta\_{0}} = \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)})-y^{(i)})x\_{j}^{(i)} \qquad \text{for } j = 0 \\]
\\[ \frac{\partial J(\theta)}{\partial \theta\_{j}} = \left( \frac{1}{m} \sum\_{i=1}^{m}(h\_{\theta}(x^{(i)})-y^{(i)})x\_{j}^{(i)} \right) + \frac{\lambda}{m} \theta\_{j} \qquad \text{for } j \geq 1 \\]

My solution uses `[0; ones(size(theta, 1) - 1, 1)]` to filter the theta values except \\( \theta\_{0} \\):

	function [J, grad] = costFunctionReg(theta, X, y, lambda)
		m = length(y); % number of training examples
		h = sigmoid(X * theta);
		J = (1/m) * (-y' * log(h) - (1 - y') * log(1 - h)) + lambda / 2 / m * (theta' .* [0; ones(size(theta, 1) - 1, 1)]') * theta;
		grad = (1/m) * (X' * (h - y)) + lambda / m * theta .* [0; ones(size(theta, 1) - 1, 1)];
	end

This solution first compute \\( J(\theta) \\) and gradient without regularization and then add the regularization term using filter `[0; theta(2:end)]` for theta:

	[J, grad] = costFunction(theta, X, y);

	% Deal with the theta(1) term
	thetaFiltered = [0; theta(2:end)];
	
	% J is the the non-regularized cost plus regularization
	J = J + ((lambda / (2*m)) * (thetaFiltered' * thetaFiltered));
	
	% grad is the non-regularized cost plus regularization.
	grad = grad + ((lambda / m) * thetaFiltered);

### Plotting the decision boundary in `plotDecisionBoundary.m` ###

`plotDecisionBoundary.m` plots the non-linear decision boundary by computing the classifier’s predictions on an evenly spaced grid and then and drew a contour plot of where the predictions change from \\( y = 0 \\) to \\( y = 1 \\).

	function plotDecisionBoundary(theta, X, y)
		plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
	
	    % Calculate the decision boundary line
	    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
	
	    % Plot, and adjust axes for better viewing
	    plot(plot_x, plot_y)
	    
	    % Legend, specific for the exercise
	    legend('Admitted', 'Not admitted', 'Decision Boundary')
	    axis([30, 100, 30, 100])
	end

