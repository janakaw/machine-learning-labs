% Set params
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('./data/neural-nw-data.mat');
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

pause;

load('./data/nn-weights.mat');

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

%% Compute cost (feedforward)
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

pause;

% Implement regularization

lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

pause;

g = sigmoidGradient([1 -0.5 0 0.5 1]);

pause;


% Initializing pameters

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Implement backpropagation

checkNNGradients;

pause;


% Implement regularization

lambda = 3;
checkNNGradients(lambda);

debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

pause;


% Training NN

options = optimset('MaxIter', 50);

lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pause;

displayData(Theta1(:, 2:end));

pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nneural nw training set accuracy: %f\n', mean(double(pred == y)) * 100);
