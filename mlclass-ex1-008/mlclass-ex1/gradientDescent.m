function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
% input 
% X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
% theta = zeros(2, 1); % initialize fitting parameters

% Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %
        % need to transpose X to meet the lecture format 
        s = zeros(length(theta), 1);
        for i = 1:m,
            s = s + (theta'*X(i, :)'-y(i))*X(i, :)';
        end 
        theta = theta - alpha*1/m*s;
        % ============================================================

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
    end

end
