%% Configuration file: 10 arms
% assign values to constants
d = 4;
C1 = 1;
C2 = 1;
C3 = 1;
C4 = 1;
L = 1.1;
lambda = 1;
delta = 0.1;
sigmaMax2 = 2*L;
Mf = 1;
alpha2 = sqrt(2*d*L*L*(1+log(C1/delta)/C2*log(C1/delta)/C2));
alpha3 = sqrt(2*d*L*log(d/delta));

% for arm context distribution
numActions = 10;
actionMus = {};
actionSigmas = {};
actionMus{1} = [1, 5];
actionMus{2} = [2, 6];
actionMus{3} = [3, 7];
actionMus{4} = [4, 8];
actionMus{5} = [5, 9];
actionMus{6} = [6, 10];
actionMus{7} = [7, 15];
actionMus{8} = [8, 20];
actionMus{9} = [2, 10];
actionMus{10} = [10, 20];

actionSigmas{1} = [10 5; 5 3];
actionSigmas{2} = [9 6; 6 5];
actionSigmas{3} = [6 7; 7 10];
actionSigmas{4} = [5 8; 8 16];
actionSigmas{5} = [15 9; 9 7];
actionSigmas{6} = [16 10; 10 15];
actionSigmas{7} = [10 11; 11 15];
actionSigmas{8} = [12 6; 6 5];
actionSigmas{9} = [7 4; 4 6];
actionSigmas{10} = [4 3; 3 4];

% for user context distribution
userMu = [12 23];
userSigma = [10 7; 7 5];

% for context distribution
sContexts = [];
sRewards = [];

% for parameters of reward distribution
theta = [0.6 0.3 0.2 0.65];
phi = [0.2 0.4 0.8 0.3];

% total number of users
T = 50000;
% number of round for start-up
initialRounds = 5;
% total number of interactions used in calculation
K = 4.5;
