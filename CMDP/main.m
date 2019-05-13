clear all;
clc;
tic;

MDPtoolbox_path = './MDPtoolbox/MDPtoolbox';
addpath(MDPtoolbox_path);
% for reproducibility
rng default

% parameter values
L = 2;
d = 4;
delta = 0.1;
discount = 0.99;
epsilon = 100; % note this is not used in hrucb
H = 100; % note this is not used in hrucb, needs discussion with ping-chun how to choose it
S = 2;
A = 2;
% m = ceil(128*(S*log(2)+log(S*A/delta))*H^2/(epsilon^2));
m = 100;
contextMu = zeros(d,1);
contextSigma = eye(d);
theta = [0.6 0.5 0.5 0.3];
phi = [0.5 0.2 0.8 0.9];
numStates = 2;
T = 30000;
case_number = 2;
switch case_number
    case 1
        context_mode = "continuous";
        numSamples = 30000;
        numPartition = 50;
        numActions = 2;
    case 2
        context_mode = "discrete";
        numSamples = 30000;
        numPartition = 10;
        numActions = 2;
end
%context_mode = "continuous";
% numSamples = T * numActions;
% state = 1 corresponds to live
% state = 2 corresponds to die
% n(s,s',a) the number of transitions from s to s' given a
% n(s,a) the number of times taking action a under state s
% n(s,a)<m indicates the (s,a) pair is not known
% n(s,a)>=m for all a indicates s is known
% r(s,a) the accumulated rewards of taking action a under state s
collectNssa = cell(numPartition, 1); % each n(s,s',a) is a 2 by 2 by 20 matrix
collectNsa = cell(numPartition, 1); % each n(s,a) is a 2 by 20 matrix
collectRsa = cell(numPartition, 1); % each r(s,a) is a 2 by 20 matrix

for i = 1:numPartition
    collectNssa{i} = zeros(numStates, numStates, numActions);
    collectNsa{i} = zeros(numStates, numActions);
    collectRsa{i} = zeros(numStates, numActions);
end

firstTime = 1;
centroids = context_space_partition(firstTime, d, numActions, numPartition, numSamples);

% tentatively will be saved as comparison with the hrucb
allContexts = zeros(T,d*numActions);
userRegrets = zeros(T,1);
mdpIdx = 1;
%k = 4;
for episode = 1:T
%     beta = generate_beta();
    if mod(episode,500) == 0
        fprintf('User %d\n', episode);
    end
    %if episode >= 9500
    %    fprintf('Stop here...\n');
    %end
    beta = 0;
    m = 1*power(episode, 0.5);
    contexts = cell(numActions,1);
    sampleContext = zeros(1,d*numActions);
    if strcmp(context_mode, "discrete")
        mdpIdx = randi([1 numPartition]);  
        for j = 1:numActions
            x = centroids(mdpIdx,4*j-3:4*j);
            x = x/norm(x);
            sampleContext(4*j-3:4*j) = x;
            contexts{j} = x;
        end
    else
        for j = 1:numActions
%         centerIdx = randi(numPartition);
%         centroidUsed = centroids(centerIdx, :);
%         x = centroidUsed(4*j-3:4*j) + mvnrnd(contextMu, contextSigma*0.00001, 1);
            x = (mvnrnd(contextMu, contextSigma, 1));
            x = x/norm(x);
            sampleContext(4*j-3:4*j) = x;
            contexts{j} = x;
        end
        mdpIdx = find_partition(sampleContext, centroids);
    end


    allContexts(episode,:) = sampleContext;
    
    rewardMus = cell(numActions,1);
    rewardSigmas = cell(numActions,1);
    for j = 1:numActions
        rewardMus{j} = dot(contexts{j}, theta);
        rewardSigmas{j} = dot(contexts{j},phi)+L;
    end
    armOptimal = optimalPolicy(beta, theta, phi, L, contexts);
    
    
    %mdpIdx = find_partition(sampleContext, centroids);
    Nssa = collectNssa{mdpIdx};
    Nsa = collectNsa{mdpIdx};
    Rsa = collectRsa{mdpIdx};
    [P, R] = PR_construction(numStates, numActions, m, Nssa, Nsa, Rsa);
    fprintf(mdp_check(P, R));
    % policy a numStates by 1 vector
    % it stores the index of best action at the row corresponding state
    [V, policy] = mdp_policy_iteration(P, R, discount);
%     other candidates for policy
%     [policy] = mdp_value_iteration(P, R, discount);
%     [V, policy] = mdp_LP(P, R, discount);
%     [~, V, policy] = mdp_Q_learning(P, R, discount);
    
    
    armChosen = contextual_mdp(policy, Nsa, m, numActions);
    
    live = true;
        
    while(live == true)
        observedReward = mvnrnd(rewardMus{armChosen},rewardSigmas{armChosen},1);
        if observedReward < beta
            live = false;
        else
            live = true;
        end
        %Nsa(1, armChosen) < m
        if true 
            Nsa(1, armChosen) = Nsa(1, armChosen) + 1;

            if live == true
                Rsa(1, armChosen) = Rsa(1, armChosen) + 1;
                Nssa(1, 1, armChosen) = Nssa(1, 1, armChosen) + 1;
            else
                Rsa(1, armChosen) = Rsa(1, armChosen) + 0;
                Nssa(1, 2, armChosen) = Nssa(1, 2, armChosen) + 1;
            end
        end
    end
    
    % update the statistics based on the observed data
    collectNssa{mdpIdx} = Nssa;
    collectNsa{mdpIdx} = Nsa;
    collectRsa{mdpIdx} = Rsa;
    
    mu = 0;
    sigma = 1;
    pd = makedist('Normal','mu',mu,'sigma',sigma);
    x_opt = contexts{armOptimal};       
    normalizedX = (beta - dot(theta,x_opt))/sqrt(get_f_value(phi,x_opt,L));
    %accumOptimalRewards = dot(theta,x_opt)*(cdf(pd,normalizedX))^(-1);
    accumOptimalRewards = (cdf(pd,normalizedX))^(-1);

    x_emp = contexts{armChosen};
    normalizedX_emp = (beta - dot(theta,x_emp))/sqrt(get_f_value(phi,x_emp,L));
    %accumRewards = dot(theta,x_emp)*(cdf(pd,normalizedX_emp))^(-1);
    accumRewards = (cdf(pd,normalizedX_emp))^(-1);

    individualRegret = accumOptimalRewards - accumRewards;
    userRegrets(episode) = individualRegret;
end
cumulativeRegret = cumsum(userRegrets);
plot(cumulativeRegret);

toc;