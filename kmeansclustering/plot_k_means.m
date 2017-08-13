function plot_k_means(X, centroids, previous, idx, k, i)

%scatter plot
palette = hsv(k + 1);
colors = palette(idx, :);

scatter(X(:,1), X(:,2), 15, colors);


% Mark centroids
plot(centroids(:,1), centroids(:,2), 'x', ...
     'MarkerEdgeColor','k', ...
     'MarkerSize', 10, 'LineWidth', 3);

% Draw centroids paths
for j=1:size(centroids,1)
    p1 = centroids(j, :);
    p2 = previous(j, :);
    plot([p1(1) p2(1)], [p1(2) p2(2)]);
end

% Title
title(sprintf('Iteration number %d', i))

end

