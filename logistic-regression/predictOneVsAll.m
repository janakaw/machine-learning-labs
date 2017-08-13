function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

z = all_theta*X';
g = sigmoid(z);
[v i] = max(g);
p = i';

end
