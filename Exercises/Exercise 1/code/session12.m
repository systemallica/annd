clear all
clc
close all

%%%%%%%%%%%
% A script comparing performance of 'trainlm', 'trainbfg' and 'traingd'
% trainlm - Levenberg-Marquardt
% trainbfg - Quasi-Newton backpropagation 
% traingd - Gradient Descent 
%%%%%%%%%%%

%generation of examples and targets
x=0:0.05:6*pi; y=sin(x.^2);
% add noise to the data to see its effect on the accuracy
y = y + 0.25 * rand(size(y));
% convert the data to a useful format
p=con2seq(x); 
t=con2seq(y); 

r1s = [];
r2s = [];
for i = 1:10
    disp(i);
    %creation of networks
    net1=feedforwardnet(20,'trainlm');
    net2=feedforwardnet(20,'trainbr');

    %set the same weights and biases for the networks
    net2.iw{1,1}=net1.iw{1,1};   
    net2.lw{2,1}=net1.lw{2,1};
    net2.b{1}=net1.b{1};
    net2.b{2}=net1.b{2};

    %training and simulation
    net1.trainParam.epochs=1000;
    net2.trainParam.epochs=1000;
    net1=train(net1,p,t);
    net2=train(net2,p,t);
    a13=sim(net1,p); a23=sim(net2,p);

    [a1, b1, r1] = postregm(cell2mat(a13),y);
    close all
    [a2, b2, r2] = postregm(cell2mat(a23),y);
    close all
    r1s = [r1s; r1];
    r2s = [r2s; r2];
end

mean1 = mean(r1s);
mean2 = mean(r2s);
disp(mean1);
disp(mean2);
% plots
% figure
% subplot(3,1,1);
% plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
% title('950 epochs');
% legend('target','trainlm','trainbr','Location','north');
% subplot(3,1,2);
% postregm(cell2mat(a13),y);
% subplot(3,1,3);
% postregm(cell2mat(a23),y);
