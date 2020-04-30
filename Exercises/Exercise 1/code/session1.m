% PERCEPTRON-> 
%   input can be any value, 
%   output is either 0 or 1
%   simple classification tool

% P is an array which contains 4 points: (2,2) (1,-2) (-2,2) (-1, 1)
P = [2 1 -2 -1; 2 -2 2 1];
% T contains the class of each point
T = [0 1 0 1];
% plot perceptron input/target vectors
plotpv(P,T);

% create perceptron
net = newp(P,T,'hardlim','learnp');
% initialize it, all weights and biases are 0 by default
net = init(net);

% set number of training iterations(epochs)
net.trainParam.epochs = 20;
% trains the perceptron and returns it and a description of the learning
% process
[net,tr_descr] = train(net,P,T);
% plot the classification performed by the perceptron
plotpc(net.IW{1},net.b{1});

% simulate the network on a new data point
Pnew = [1;-0.3];
sim(net,Pnew)

