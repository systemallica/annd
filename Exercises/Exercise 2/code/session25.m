% formatting data
if istable(lasertrain), lasertrain=table2array(lasertrain); end
if istable(laserpred), laserpred=table2array(laserpred); end
% first param: dataset
% second param: training lag (t-5)
% trainData contains the target, delayed by 1, 2, 3, 4 and the 5 t
lags = [100];
hidden = [40];
error = [];
minError = 999;
for i = 1:length(lag)
    
    lag = lags(i);
    [trainData, trainTarget] = getTimeSeriesTrainData(lasertrain, lag);
    
    disp(length(trainData));
    disp(length(trainTarget));
    
    for j = 1:length(hidden)
       
        for k = 1:10
            % 1st: train feedforward net
            net=feedforwardnet(hidden(j),'trainbr');

            net.trainParam.epochs=100;

            net.divideFcn='dividetrain';
            net.divideParam.trainRatio = 100/100;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;

            net=init(net);
            net=train(net,trainData,trainTarget);   

            % initialize output
            predictedTargets = [];

            for z=1:(size(laserpred,1))
                % predict value using the last n points from the training dataset
                % n = lag
                % use this to predict the next value
                % modify the input vector with the new data

                % iterate 100 times

                % take the last known trainData column, take the las 4 values + the
                % last known trainTarget
                if z == 1
                    data = [trainData(2:lag,size(trainData,2)-1+z); trainTarget(size(trainData,2)-1+z)];
                else
                    data = [data(2:lag); predictedTarget];
                end

                % predict next point
                predictedTarget = net(data);

                % add predicted point to array
                predictedTargets = [predictedTargets; predictedTarget];    
            end
            
            mse = sqrt(mean((laserpred-predictedTargets).^2));
            
            if(mse < minError)
                minError = mse;
                bestTargets = predictedTargets;
            end
            error = [error; mse];
            
            close all
            plot(predictedTargets);
            hold on;
            plot(laserpred);
            legend('predicted', 'real');
        end
    end
end

disp(minError);

close all
plot(bestTargets);
hold on;
plot(laserpred);
legend('predicted', 'real');