function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(size(X,1), 1) X];
z1= Theta1*X';
z2 = sigmoid(z1);
g2 = [ones(size(z2',1), 1) z2'];
z3 = Theta2*g2';
[v index] = max(sigmoid(z3));
p = index';

end
