function [Y_Pred,rmse]=LSTM(lasertrain, laserpred, LAG)

%% Standardize training data
mu = mean(lasertrain);
sig = std(lasertrain);

[XData, YData] = getTimeSeriesTrainData(lasertrain, LAG);

X_Train = (XData - mu) ./ sig;
Y_Train = (YData - mu) ./ sig;

%% create LSTM regression network
numFeatures = LAG;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% set the options
options = trainingOptions('adam', ...
    'MaxEpochs',350, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.05, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.09, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% train the network
net = trainNetwork(X_Train,Y_Train,layers,options);
 
%% predict future point

% init net
net = predictAndUpdateState(net,X_Train);

% get first prediction
x_points = X_Train(:,end);
x_points = [x_points(2:end, end); Y_Train(:,end)];
[net,Y_Pred] = predictAndUpdateState(net,x_points);

x_points = [x_points(2:end, end); Y_Pred];

% predict next 99 points
for i = 2:100
    [net,Y_Pred(:,i)] = predictAndUpdateState(net,x_points,'ExecutionEnvironment','cpu');
    x_points = [x_points(2:end, end); Y_Pred(:,i)];
end
 
%% unstandardize the data
Y_Pred = sig*Y_Pred + mu;
 
%% calculate error
rmse = sqrt(mean((Y_Pred'-laserpred).^2));