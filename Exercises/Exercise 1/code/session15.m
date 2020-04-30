%%%%%%%%%%%
% A script comparing performance of 'trainlm' and 'traingd'
% trainlm - Levenberg-Marquardt
% trainbr - Bayesian Regularization Backpropagation
%%%%%%%%%%%


%generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2);
% add noise to the data to see its effect on the accuracy
% y = y + 0.1 * rand(size(y));
% convert the data to a useful format
p=con2seq(x); 
t=con2seq(y); 

%creation of networks
net1=feedforwardnet(30,'trainlm');
net2=feedforwardnet(30,'trainbfg');
net3=feedforwardnet(30,'trainbr');
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

net1.trainParam.epochs=10;
net2.trainParam.epochs=10;
net3.trainParam.epochs=10;
net1=train(net1,p,t);
net2=train(net2,p,t);
net3=train(net3,p,t);
a12=sim(net1,p); a22=sim(net2,p); a32=sim(net3,p);

net1.trainParam.epochs=30;
net2.trainParam.epochs=30;
net3.trainParam.epochs=30;
net1=train(net1,p,t);
net2=train(net2,p,t);
net3=train(net3,p,t);
a13=sim(net1,p); a23=sim(net2,p); a33=sim(net3,p);

%plots
figure
subplot(3,3,1);
plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a31),'b'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainlm','trainbr','Location','north');
subplot(3,3,2);
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
%subplot(3,4,3);
%postregm(cell2mat(a21),y);
subplot(3,3,3);
postregm(cell2mat(a31),y);
%
subplot(3,3,4);
plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a32),'b');
title('10 epochs');
legend('target','trainlm','trainbr','Location','north');
subplot(3,3,5);
postregm(cell2mat(a12),y);
%subplot(3,4,7);
%postregm(cell2mat(a22),y);
subplot(3,3,6);
postregm(cell2mat(a32),y);
%
subplot(3,3,7);
plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a33),'b');
title('30 epochs');
legend('target','trainlm','trainbr','Location','north');
subplot(3,3,8);
postregm(cell2mat(a13),y);
%subplot(3,4,11);
%postregm(cell2mat(a23),y);
subplot(3,3,9);
postregm(cell2mat(a33),y);
