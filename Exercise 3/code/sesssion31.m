% 50x500 matrix with random Gaussian numbers
% = 500 datapoints of dimension 50
p = 50;
X = randn(p,500);
% calculate covariance matrix of dataset X
covMatrix = cov(X');
% returns q largest eigenvalues in the matrix(D), and the p×q matrix(V) 
% with the corresponding eigenvectors of the square matrix
q = 20;
[eigenvectors,eigenvalues]=eigs(covMatrix,q);
eigenvalues = diag(eigenvalues);

reducedDataset = (X' * eigenvectors)';

reconstructedDataset = (eigenvectors * reducedDataset) + mean(X);

% estimation of the error
error = sqrt(mean(mean((X-reconstructedDataset).^2)));