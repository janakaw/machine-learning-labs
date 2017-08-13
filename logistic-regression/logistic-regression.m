%% Logistic Regression

%% setup parameters 
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('./data/logistic_regression_data.mat');
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

pause;

%% Logistic Regression
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

pause;

% Predict
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

