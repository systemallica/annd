% Each line of this matrix is a single 16 by 16 image of a handwritten 
% 3 that has been expanded out into a 256 longvector.
load threes -ASCII;

%% average three
avg_three = mean(threes);
colormap('gray');
imagesc(reshape(avg_three(1,:),16,16),[0,1]);

%% some PCA results
% PCA 1 component
[eigenvalues1, reducedDataset1, reconstructedDataset1] = PCA(1,threes);
% PCA 2 component
[eigenvalues2, reducedDataset2, reconstructedDataset2] = PCA(2,threes);
% PCA 3 component
[eigenvalues3, reducedDataset3, reconstructedDataset3] = PCA(3,threes);
% PCA 4 component
[eigenvalues4, reducedDataset4, reconstructedDataset4] = PCA(4,threes);
% PCA 5 component
[eigenvalues5, reducedDataset5, reconstructedDataset5] = PCA(5,threes);
% PCA 256 component
[eigenvalues, reducedDataset, reconstructedDataset] = PCA(256,threes);

%% plot the eigenvalues of the whole dataset
plot(eigenvalues);
axis([-3 260 -1 25]);
title('Full dataset eigenvalues');

%% Plot some numbers from reconstructed PCA datasets
colormap('gray');
subplot(3,2,1);
imagesc(reshape(reconstructedDataset1(45,:),16,16),[0,1]);
title('PCA 1 component');
subplot(3,2,2);
imagesc(reshape(reconstructedDataset2(45,:),16,16),[0,1]);
title('PCA 2 components');
subplot(3,2,3);
imagesc(reshape(reconstructedDataset3(45,:),16,16),[0,1]);
title('PCA 3 components');
subplot(3,2,4);
imagesc(reshape(reconstructedDataset4(45,:),16,16),[0,1]);
title('PCA 4 components');
subplot(3,2,5);
imagesc(reshape(reconstructedDataset5(45,:),16,16),[0,1]);
title('PCA 5 components');
subplot(3,2,6);
imagesc(reshape(threes(45,:),16,16),[0,1]);
title('Original dataset');

%% error vs cumeigenvalues
% Calculate errors for PCAs from 1 to 50 components
errors = [];
for i=1:50
    [eigenvalues, reducedDataset, reconstructedDataset] = PCA(i,threes);
    error = sqrt(mean(mean((threes-reconstructedDataset).^2)));
    errors = [errors; error];
end

% error for w=256 should be 0 but it's 0.3 because of the lost info between
% conversions

clf
subplot(2,1,1);
plot(errors);
title('RMSE');

cumeigenvalues = [];
for i=1:50
    cumeigenvalues = [cumeigenvalues; cumsum(eigenvalues(i:end))];
end

subplot(2,1,2);
plot(cumeigenvalues);
title('Cumulative eigenvalues');