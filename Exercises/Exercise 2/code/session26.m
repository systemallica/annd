close all

%% formatting data
if istable(lasertrain), lasertrain=table2array(lasertrain); end
if istable(laserpred), laserpred=table2array(laserpred); end

%% define lag constant
LAG = 100;

%% predict next 100 points
[Y_Pred, rmse] = LSTM(lasertrain, laserpred, LAG);
disp(rmse);

%% plot results
plot(Y_Pred);
hold on;
plot(laserpred);
legend('predicted', 'real');