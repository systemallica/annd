% p = 21
load choles_all;
% with two eigenvalues we get almost the same error as with 10, as the two
% first eigenvectors cover most of the variance of the data
q = 2;

[eigenvalues, reducedDataset, reconstructedDataset] = PCA(q,p);

% estimation of the error
error = sqrt(mean(mean((p-reconstructedDataset).^2)));