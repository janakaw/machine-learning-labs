%% support vector machine to classify haberman surival stats dataset
% Attribute Information:
%   1. Age of patient at time of operation (numerical)
%   2. Patient's year of operation (year - 1900, numerical)
%   3. Number of positive axillary nodes detected (numerical)
%   4. Survival status (class attribute)
%         1 = the patient survived 5 years or longer
%         2 = the patient died within 5 year
%

%% Load and visualize data (only first two attributes are used)
fprintf('Loading Data..\n')

Z = load('./data/haberman.data');
X = Z(:,[1,2]);
y = Z(:, [4]);
y = y - 1;

plotData(X, y);

fprintf('paused, \n');
pause;

%% Training Linear SVM

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('paused, \n');
pause;

%% Training SVM with RBF Kernel

fprintf('\nTraining with RBF Kernel \n');

% SVM Parameters
C = 1; sigma = 0.1;

% Set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, we will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('paused, \n');
pause;
