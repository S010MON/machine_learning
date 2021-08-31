function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

	for i = 1:num_iters
		
		% Calculate the derivative
		delta = (X' * (X * theta - y))/m;
		
		% Update theta
		theta = theta - (alpha * delta);

    		% Save the cost J in every iteration    
    		J_history(i) = computeCost(X, y, theta);
	end%for
end%function
