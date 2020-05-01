% attractors are handwritten digits 0..9
%   noise: represents the level of noise that will corrupt the digits and is a number between 0 and 10
%   numiteris: the number of iterations the Hopfield network (having as input the noisy digits) will run
noise = 5;
numiter = 1000;

hopdigit_v2(noise,numiter);

% can it always reconstruct the noisy digits?

% Not always capable, with a noise factor = 5, it is already confusing 3
% with 5 and 5 with 8. also 9 with 2

% Increasing the number of iterations does help at it usually converges to
% the right solution

% When the noise factor is really high, not even a high number of
% iterations gets the correct solution