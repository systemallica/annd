%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n=8;

a = [1 -1; 0.5 -0.5; 0.3 -0.3; 0.6 -0.6; -0.7 0.7; -0.5 0.5; 0.1 -0.1; 0.2 -0.2]';

for i=1:n 
    b = [a(1,i); a(2,i)];
    [y,Pf,Af] = sim(net,{1 50},{},b);   % simulation of the network for 50 timesteps 
    record = [b cell2mat(y)];
    plot(b(1,1), b(2,1),'bx', record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');

% For very symmetric points, only two attractors are shown(those at drawn
% diagonal)
% Also, one of them is not in the initial attractors
% So no, they can be different from the ones stored at the beginning