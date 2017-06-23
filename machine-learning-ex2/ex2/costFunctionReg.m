function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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



equation=X*theta;
hx=sigmoid(equation);
log_eq=log(hx);
log_minus_one=log(1-(hx));
cost=y'*log_eq +(1.-y)'*log_minus_one;
reg_cost_matrix=theta.**2;
reg_cost=sum(reg_cost_matrix)-reg_cost_matrix(1);
reg_cost=reg_cost*lambda/2;
J=(-cost+reg_cost)/m;

for iter=1:size(theta)
p=(hx-y)'*X(:,iter);
if(iter!=1)
p=p+lambda*theta(iter);
endif
grad(iter)=p;

end
grad=grad./m;




% =============================================================

end
