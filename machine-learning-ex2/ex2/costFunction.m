function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


equation=X*theta;
hx=sigmoid(equation);
log_eq=log(hx);
log_minus_one=log(1-(hx));
cost=y'*log_eq +(1.-y)'*log_minus_one;
J=-cost./m;

for iter=1:size(theta)
p=(hx-y)'*X(:,iter);
grad(iter)=p;

end
grad=grad./m;

% =============================================================

end
