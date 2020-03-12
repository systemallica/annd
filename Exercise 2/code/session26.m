% formatting data
if istable(lasertrain), lasertrain=table2array(lasertrain); end
if istable(laserpred), laserpred=table2array(laserpred); end

lasertrain = lasertrain';

% standardize data
mu = mean(lasertrain);
sig = std(lasertrain);

dataTrainStandardized = (lasertrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

% create LSTM regression network
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% set the options
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% train the network
net = trainNetwork(XTrain,YTrain,layers,options);

% predict future points
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = 100;
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

% unstandardize the data
YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-laserpred).^2));

plot(YPred);
hold on;
plot(laserpred);
legend('predicted', 'real');