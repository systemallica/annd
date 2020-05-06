% p = 21
load choles_all;
% with two eigenvalues we get almost the same error as with 10, as the two
% first eigenvectors cover most of the variance of the data

rmses = [];
for q=1:21
    [eigenvalues, reducedDataset, reconstructedDataset] = PCA(q,p);

    % estimation of the error
    rmse = sqrt(mean(mean((p-reconstructedDataset).^2)));
    rmses = [rmses; rmse];
end

plot(rmses);
xlabel('PCA dimensions');
ylabel('RMSE');
