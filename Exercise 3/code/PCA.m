function [eigenvalues, reducedDataset, reconstructedDataset] = PCA(q,X)
% PCA 
% calculate covariance matrix of dataset X
covMatrix = cov(X');
% returns q largest eigenvalues in the matrix(D), and the p×q matrix(V) 
% with the corresponding eigenvectors of the square matrix
[eigenvectors,eigenvalues]=eigs(covMatrix,q);
eigenvalues = diag(eigenvalues);

reducedDataset = (X' * eigenvectors)';

reconstructedDataset = (eigenvectors * reducedDataset) + mean(X);
end

