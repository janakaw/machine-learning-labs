function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

I_mat = eye(num_labels, num_labels);

yy = [];
for i = 1:size(y, 1)
  yy = [yy; I_mat(y(i,1), :);];
end

X = [ones(size(X, 1), 1) X];

Z1 = X*Theta1';
g1 = sigmoid(Z1);

g1 = [ones(size(g1,1), 1) g1];

Z2 = g1*Theta2';
g2 = sigmoid(Z2);

log_h1 = log(g2);
log_h2 = log(1 - g2);

for i = 1:m
  for j = 1:size(g2, 2)
    J += (-1*yy(i, j)*log_h1(i,j) - (1 - yy(i, j))*log_h2(i,j));     
  end
end

J = J/m;
 
Theta11 = Theta1;
Theta11(:, [1]) = []; 

Theta21 = Theta2;
Theta21(:, [1]) = []; 
s1 = sum(sum(Theta21.*Theta21));
s2 = sum(sum(Theta11.*Theta11));
J = J + (lambda/(2*m))*(s1 + s2)

d3 = g2 - yy;

TTheta2 = Theta2(:, 2:end);

d2 = d3*TTheta2 .* sigmoidGradient(Z1);
delta1 = d2'*X;
delta2 = d3'*g1;

  Theta1_grad = (1/m)*delta1;
  Theta2_grad = (1/m)*delta2;
  
t1_gc1 = Theta1_grad(:, 1);
t2_gc1 = Theta2_grad(:, 1);

t1_grem = Theta1_grad(:, 2:end);
t2_grem = Theta2_grad(:, 2:end);
  
t1_grem = t1_grem + (lambda/m)*Theta1(:, 2:end);
t2_grem = t2_grem + (lambda/m)*Theta2(:, 2:end);

Theta1_grad = [t1_gc1 t1_grem];
Theta2_grad = [t2_gc1 t2_grem];   

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
