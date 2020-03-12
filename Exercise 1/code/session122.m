% r-number: 0767329
% 5 largest digits in descending order: 97763
% build new target Tnew ---------------------------------------------------
Tnew = (9 * T1 + 7 * T2 + 7 * T3 + 6 * T4 + 3 * T5)/(9 + 7 + 7 + 6 + 3);

% Create subsets ----------------------------------------------------------
% Training set
% Subset of 3000 random samples 
[tSubset,i1] = datasample(Tnew, 3000, 'Replace', false);
xSubset = X1(i1);
ySubset = X2(i1);

% Create the different subsets
% Training 
tTraining = tSubset(1:1000);
xTraining = xSubset(1:1000);
yTraining = ySubset(1:1000);
% Validation 
tValidation = tSubset(1001:2000);
xValidation = xSubset(1001:2000);
yValidation = ySubset(1001:2000);
% Test 
tTest = tSubset(2001:3000);
xTest = xSubset(2001:3000);
yTest = ySubset(2001:3000);

% Plot the obtained training subset ---------------------------------------
% scatteredInterpolant: creates an interpolant of the subset
% it is an approximation of the function we want to find t(i) = f(x(i),y(i))
interpolant = scatteredInterpolant(xTraining, yTraining, tTraining);
% linspace: creates a linearly spaced vector
% meshgrid: creates a cartesian grid from two vectors
%   if the vector has size 100, it will return a matrix 100x100 with the
%   values of the vector in each row
[X,Y] = meshgrid(linspace(0,1,1000), linspace(0,1,1000));
% draw a mesh of the two matrices X and Y and the value given by the
% interpolant for the points of those matrices
% this changes every time due to sampling(the mesgrid stays constant cause
% we're creating it from a linspace)
mesh(X,Y,interpolant(X,Y));

% Training the net --------------------------------------------------------
neurons = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 8, 5, 3, 1];
algorithms = ["traingd", "traingda", "traincgf", "traincgp", "trainbfg", "trainlm"];

% test all combinations of neurons and algorithms to fine tune the
% parameters of the ANN

% initialize variables
bestR = 0;
bestNumberOfNeurons = 101;
bestAlgorithm = '';
bestNet = 0;

for i = 1:length(algorithms)
    for j = 1:length(neurons)
        % Create the ANN
        net = feedforwardnet(neurons(j), algorithms(i));
        % we need to transpose the vectors
        pTraining = [xTraining, yTraining]';
        % set the max number of epochs for the training(stops early if it converges
        % before)
        net.trainParam.epochs=1000;   
        % train the network
        net=train(net,pTraining,tTraining');   
        
        % Use validation set to find the best model
        pValidation = [xValidation, yValidation]';
        a1=sim(net,pValidation); 
        [m,b,r] = postregm(a1, tValidation');
        % If r-value is higher and number of neurons is <= than the
        % previous solution, store number of neurons and algorithm
        if r >= bestR && neurons(j) < bestNumberOfNeurons
            bestR = r;
            bestNumberOfNeurons = neurons(j);
            bestAlgorithm = algorithms(i);
            bestNet = net;
        end        
    end
end

disp(bestR)
disp(bestNumberOfNeurons)
disp(bestAlgorithm)  
disp(bestNet)

% Use test set to check the results 
pTest = [xTest, yTest]';
tTest = tTest';
a2=sim(bestNet,pTest);

% plot the results
% 3D-plot of the predicted values and the expected values
close all
figure
subplot(1,2,1)
postregm(a2, tTest);
subplot(1,2,2)
plot3(xTest, yTest, tTest, '.', 'Color', 'b');
hold on;
plot3(xTest, yTest, a2, '.', 'Color', 'r');

% mesh of the error of the prediction
% error = expected value - predicted value

% mean squared error of the predictions
mse = immse(tTest, a2);
disp(mse)

