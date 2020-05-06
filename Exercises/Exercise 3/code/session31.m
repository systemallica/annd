% 50x500 matrix with random Gaussian numbers
% = 500 datapoints of dimension 50
p = 50;
X = randn(p,500);


rmses = [];
for q=1:50
    [eigenvalues, reducedDataset, reconstructedDataset] = PCA(q,X);

    % estimation of the error
    rmse = sqrt(mean(mean((X-reconstructedDataset).^2)));
    rmses = [rmses; rmse];
end

plot(rmses);
xlabel('PCA dimensions');
ylabel('RMSE');