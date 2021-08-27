function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

	m = length(y); 				% number of training examples
	h_theta = sum(theta'.*X, 2);		% calculate each h(X) [sum axis 2 = x0+x1+..xn]
	J = (1/(2*m)) * sum((h_theta - y).^2);  % return sum of the squared differences

end%function
