% 50x500 matrix with random Gaussian numbers
% = 500 datapoints of dimension 50
p = 50;
X = randn(p,500);
q = 30;

[eigenvalues, reducedDataset, reconstructedDataset] = PCA(q,X);

% estimation of the error
error = sqrt(mean(mean((X-reconstructedDataset).^2)));