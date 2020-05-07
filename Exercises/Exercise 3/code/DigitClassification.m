close all
nntraintool('close');
nnet.guis.closeAllViews();
echo off;

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
rng('default')


% Load the training data into memory
load('digittrain_dataset.mat');

% Layer 1
% hiddenSize1 = 100;
% autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
%     'MaxEpochs',400, ...
%     'L2WeightRegularization',0.004, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.15, ...
%     'ScaleData', false);
% 
% figure;
% plotWeights(autoenc1);
% feat1 = encode(autoenc1,xTrainImages);
% 
% % Layer 2
% hiddenSize2 = 50;
% autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% 
% feat2 = encode(autoenc2,feat1);
% 
% % Layer 3
% softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
% 
% % Deep Net
% deepnet = stack(autoenc1,autoenc2,softnet);
% view(deepnet);
 
% Test deep net
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
load('digittest_dataset.mat');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
% y = deepnet(xTest);
% figure;
% plotconfusion(tTest,y);
% classAcc1=100*(1-confusion(tTest,y))
% pause; %83.1

% Test fine-tuned deep net
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
% deepnet = train(deepnet,xTrain,tTrain);
% y = deepnet(xTest);
% figure;
% plotconfusion(tTest,y);
% classAcc2=100*(1-confusion(tTest,y))
% view(deepnet)
% pause; % 99.7


classAcc3 = [];
classAcc4 = [];
classAcc5 = [];
for i=1:10
    %Compare with normal neural network (1 hidden layers)
    net = patternnet(100);
    net=train(net,xTrain,tTrain);
    y=net(xTest);
    plotconfusion(tTest,y);
    classAcc3=[classAcc3; 100*(1-confusion(tTest,y))]

    %Compare with normal neural network (2 hidden layers)
    net = patternnet([100 50]);
    net=train(net,xTrain,tTrain);
    y=net(xTest);
    plotconfusion(tTest,y);
    classAcc4=[classAcc4; 100*(1-confusion(tTest,y))]

    %Compare with normal neural network (3 hidden layers)
    net = patternnet([200 100 50]);
    net=train(net,xTrain,tTrain);
    y=net(xTest);
    plotconfusion(tTest,y);
    classAcc5=[classAcc5; 100*(1-confusion(tTest,y))]
end

disp(mean(classAcc3));
disp(mean(classAcc4));
disp(mean(classAcc5));
% changing number of layers and hidden units only seems to worsen the
% results

% no matter what, a finetuned encoder-decoder always works better

% a single 2000 units layer gets 74% acc

% 10 layers of 50 units get 27%

% best results are with 100 50

% it takes a longer time to train though