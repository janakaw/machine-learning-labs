function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);

J = 0;
grad = zeros(size(theta));

g = sigmoid(X*theta);
c = -1*y'*log(g) - (1 - y)'*log(1-g);
J = (1/m)*c + (lambda/(2*m))*(theta(2:end,:))'*(theta(2:end,:));
temp = theta;
temp(1) = 0;
grad = ((1/m)*(g - y)'*X)' + (lambda/m)*temp;

end
