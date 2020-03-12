% BACKPROPAGATION AND FEEDFORWARD
%P = [2 1 -2 -1; 2 -2 2 1];
%T = [0 1 0 1];
%numN = 3;
%trainAlg = traingd;
% create net
%net = feedforwardnet(numN,trainAlg);
% train it 
%net = train(net,P,T);
% simulate it
%a = sim(net,P);
% calculate regression between target and output
%[m,b,r] = postreg(a,T);

clear
clc
close all

%%%%%%%%%%%
% A script comparing performance of 'trainlm', 'trainbfg' and 'traingda'
% trainlm - Levenberg - Marquardt
% trainbfg - Quasi-Newton backpropagation 
% traingda - 
%%%%%%%%%%%

%generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2);
% add noise to the data to see its effect on the accuracy
y = y + 0.1 * rand(size(y));
% convert the data to a useful format
p=con2seq(x); 
t=con2seq(y); 

%creation of networks
net1=feedforwardnet(50,'trainlm');
net2=feedforwardnet(50,'trainbfg');
net3=feedforwardnet(50,'traingda');
%set the same weights and biases for the networks
net2.iw{1,1}=net1.iw{1,1};   
net2.lw{2,1}=net1.lw{2,1};
net3.iw{1,1}=net2.iw{1,1};
net3.lw{2,1}=net2.lw{2,1};
net2.b{1}=net1.b{1};
net3.b{1}=net2.b{1};
net2.b{2}=net1.b{2};
net3.b{2}=net2.b{2};

%training and simulation
% set the number of epochs for the training
net1.trainParam.epochs=1;   
net2.trainParam.epochs=1;
net3.trainParam.epochs=1;
% train the networks
net1=train(net1,p,t);   
net2=train(net2,p,t);
net3=train(net3,p,t);
% simulate the networks with the input vector p
a11=sim(net1,p); 
a21=sim(net2,p); 
a31=sim(net3,p); 

net1.trainParam.epochs=14;
net2.trainParam.epochs=14;
net3.trainParam.epochs=14;
net1=train(net1,p,t);
net2=train(net2,p,t);
net3=train(net3,p,t);
a12=sim(net1,p); a22=sim(net2,p); a32=sim(net3,p);

net1.trainParam.epochs=985;
net2.trainParam.epochs=985;
net3.trainParam.epochs=985;
net1=train(net1,p,t);
net2=train(net2,p,t);
net3=train(net3,p,t);
a13=sim(net1,p); a23=sim(net2,p); a33=sim(net3,p);

%plots
figure
subplot(3,4,1);
plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g',x,cell2mat(a31),'b'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainlm','traingd','traingda','Location','north');
subplot(3,4,2);
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
subplot(3,4,3);
postregm(cell2mat(a21),y);
subplot(3,4,4);
postregm(cell2mat(a31),y);
%
subplot(3,4,5);
plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g',x,cell2mat(a32),'b');
title('15 epochs');
legend('target','trainlm','traingd','traingda','Location','north');
subplot(3,4,6);
postregm(cell2mat(a12),y);
subplot(3,4,7);
postregm(cell2mat(a22),y);
subplot(3,4,8);
postregm(cell2mat(a32),y);
%
subplot(3,4,9);
plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g',x,cell2mat(a33),'b');
title('1000 epochs');
legend('target','trainlm','traingd','traingda','Location','north');
subplot(3,4,10);
postregm(cell2mat(a13),y);
subplot(3,4,11);
postregm(cell2mat(a23),y);
subplot(3,4,12);
postregm(cell2mat(a33),y);
