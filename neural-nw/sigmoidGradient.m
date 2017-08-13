function g = sigmoidGradient(z)

g = zeros(size(z));
gz = 1.0 ./ (1.0 + exp(-z)); 
g = gz .* (1 - gz);

end
