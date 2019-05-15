%% Configuration file: 20 arms
% choose algorithm
alg = 'HR-UCB';
%alg = 'LinUCB';
%alg = 'HR-UCB-fixed-sigma';

% assign values to constants
d = 4;
C1 = 1;
C2 = 1;
C3 = 1;
C4 = 1;
lambda = 1;
delta = 0.1;
Mf = 1;
L = 2;
alpha2 = sqrt(2*d*L*L*(1+log(C1/delta)/C2*log(C1/delta)/C2));
alpha3 = sqrt(2*d*L*log(d/delta));
objective = 'lifetime';
%objective = 'reward';
outcome_type = "Gaussian";
context_gen_mode = "randomly-generate-new-contexts";

% for arm context distribution
numActions = 20;

% for contexts
contextMu = zeros(d,1);
contextSigma = eye(d);

% for parameters of reward distribution
theta = [0.6 0.5 0.5 0.3];
phi = [0.5 0.2 0.8 0.9];
sigmaMax2 = 1*norm(phi) + L; % Note: contexts are drawn from a unit ball
beta_low = -1;
beta_high = 1;

% total number of users
T = 30000;
total_repeat = 20;

% number of round for start-up
initialRounds = 5;

% total number of interactions used in calculation
K = 5;

% for LinUCB
alpha = 1;
