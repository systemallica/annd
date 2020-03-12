%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);

n=8;
iterations = 250;

a = [1 -1 1; 0.5 -0.5 -0.5; 0.3 -0.3 0.3; 0.6 -0.6 0.6; -0.7 0.7 0.7; -0.5 0.5 0.5; 0.1 -0.1 0.1; 0.2 -0.2 0.2]';

for i=1:n 
    b = [a(1,i); a(2,i); a(3,i)];
    [y,Pf,Af] = sim(net,{1 iterations},{},b);   % simulation of the network for 50 timesteps 
    record = [b cell2mat(y)];
    plot3(b(1,1), b(2,1), b(3,1),'bx', record(1,:),record(2,:),record(3,:),'r'); % plot evolution
    hold on;
    plot3(record(1,iterations),record(2,iterations),record(3,iterations),'gO');  % plot the final point with a green circle
end

grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');


% The number of iterations needed to converge to the final attractors is
% much higher in the 3D space(250 vs 15)