% formatting data
if istable(lasertrain), lasertrain=table2array(lasertrain); end
if istable(laserpred), laserpred=table2array(laserpred); end
% first param: dataset
% second param: training lag (t-5)
% trainData contains the target, delayed by 1, 2, 3, 4 and the 5 t
lag = 50;
[trainData, trainTarget] = getTimeSeriesTrainData(lasertrain, lag);

% plot(predictTarget)

% 1st: train feedforward net
net=feedforwardnet(50,'trainlm');
net.trainParam.epochs=985;
net=train(net,trainData,trainTarget);   

% initialize output
predictedTargets = [];

disp(size(laserpred,1))

for i=1:(size(laserpred,1))
    % predict value using the last n points from the training dataset
    % n = lag
    % use this to predict the next value
    % modify the input vector with the new data
    
    % iterate 100 times
    
    % take the last known trainData column, take the las 4 values + the
    % last known trainTarget
    if i == 1
        data = [trainData(2:lag,size(trainData,2)-1+i); trainTarget(size(trainData,2)-1+i)];
    else
        data = [data(2:lag); predictedTarget];
    end
    
    % predict next point
    predictedTarget = net(data);

    % add predicted point to array
    predictedTargets = [predictedTargets; predictedTarget];    
end

plot(predictedTargets);
hold on;
plot(laserpred);
legend('predicted', 'real');
% 2nd: use as recurrent network

% a good prediction should find a "dip" in the plot(as shown in figure 2b)
% Time series data with a recurrent neural network
% Get the first segment and predict the next one
% Closed loop, use your own predictions and compare to real data