% T is a N×Q matrix containing Q vectors with components equal to ±1. 
% This command will create a recurrent Hopfield network with stable points being the vectors from T. 
% For a 2-neuron network with 3 attractors:
T = [1 1; -1 -1; 1 -1]';
% Create new hofield network
% Tries to assign new point to the closest attractor(pattern that we wanna recognise)-> classification
net = newhop(T);

% 1- Example inputs with its corresponding attractors
A1 = [0.3 0.6; -0.1 0.8; -1 0.5]'; % [1 1, -1 1, -1 1]'
A2 = [0.5 -0.6; -0.3 0.1; 1 0.6]'; % [1 -1, -1 1, 1 1]'
A3 = [-0.3 0.6; 0.1 0.9; 1 1]'; % [-1 1, 1 1, 1 1]'
A4 = [0.6 0.6; -0.2 0.2; -0.7 0.2]'; % [1 1, -1 1, -1 1]'
A5 = [0.4 0.7; -0.6 0.9; -1 0.1]'; % [1 1, -1 1, -1 1]'

% Multiple step iteration Hopfield network simulation
% 20 steps
Y1 = net({50}, {}, A1);
Y2 = net({50}, {}, A2);
Y3 = net({50}, {}, A3);
Y4 = net({50}, {}, A4);
Y5 = net({50}, {}, A5);

% 2- The number of attractors found by the network is at least the number of attractors used
% to create it. But there can be more because of spurious patters. This
% happens when the network converges to local minimum energy points which
% are not part of the originally defined attractors

% 3- It takes around 15 iterations to reach the attractors

