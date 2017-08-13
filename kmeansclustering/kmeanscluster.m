% Initialize k centroids by randomly picking k samples from X
function centroids = initCentroids(X, k)

  centroids = zeros(k, size(X, 2));

  randidx = randperm(size(X, 1));
  centroids = X(randidx(1:k), :);

end

% Find closeset centroid for each sample in X
function idx = assignCentroids(X, centroids)

k = size(centroids, 1);
idx = zeros(size(X,1), 1);

m = size(X,1);
minidx = zeros(k, m);

for i = 1:k
  cc = [];
  xx = [];
  cc = repmat(centroids(i, :), m, 1);

  xx = X - cc;
  xx = xx .^ 2;

  minidx(i, :) = ones(1, size(X, 2)) * xx';
endfor

[v idx] = min(minidx);

end


% Compute new k centroids for current iteration
function centroids = computeCentroids(X, idx, k)

[m n] = size(X);
centroids = zeros(k, n);

cent_fill = eye(k);
cent_id = [];
if(size(idx, 1) == 1)
idt = idx';
endif

for i = 1:m

if(size(idx, 1) == 1)
 cent_id = [cent_id; cent_fill(idt(i,1),:)];
 else
  cent_id = [cent_id; cent_fill(idx(i,1),:)];
 endif
endfor

mu = sum(cent_id, 1:size(cent_id, 2));
mu = 1 ./ mu;
p = cent_id' * X;
p = p .* mu';
centroids = p;

end


% Run num_iter rounds of k-means
function [centroids, idx] = runkMeans(X, initial_centroids,
                                      num_rounds, plot_progress)
  if ~exist('plot_progress', 'var') || isempty(plot_progress)
      plot_progress = false;
  end

  if plot_progress
      figure;
      hold on;
  end

  [m n] = size(X);
  k = size(initial_centroids, 1);
  centroids = initial_centroids;
  previous_centroids = centroids;
  idx = zeros(m, 1);

  for i=1:num_rounds
      fprintf('k-means iteration %d/%d...\n', i, num_rounds);
      if exist('OCTAVE_VERSION')
          fflush(stdout);
      end
      
      idx = assignCentroids(X, centroids);
      
      if plot_progress
          plot_k_means(X, centroids, previous_centroids, idx, k, i);
          previous_centroids = centroids;
          fprintf('Press key to continue.\n');
          pause;
      end
      
      centroids = computeCentroids(X, idx, k);
  end

  if plot_progress
      hold off;
  end

end


%% k-means clustering

fprintf('\nRunning k-means clustering on winedata dataset.\n');

% Load dataset
WD = load('./data/winedata.txt');

X = WD(:,[2, 3]);

% Set k-means parameters
k = 5;
iterations = 20;

% Iinit centroids by picking k random samples as centroids
ini_centroids = initCentroids(X, k)
pause

[centroids, idx] = runkMeans(X, ini_centroids, iterations, true);
fprintf('\nk-means done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


