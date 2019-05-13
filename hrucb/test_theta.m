%% Test the learning rate of theta vector
clear all;
clc;
tic;

N = 5000;
d = 4;
%actionMu = [1 1];
%actionSigma = [100 80; 80 100];

% for user context distribution
%userMu = [2 3];
%userSigma = [100 70; 70 100];

Mu = zeros(d,1);
Sigma = eye(d);
contexts = zeros(N, d);
rewards = zeros(N, 1);
thetaHat_history = cell(N,1);
phiHat_history = cell(N,1);
% for parameters of reward distribution
%theta = [0.7 0.4 0.2 0.5];
%phi = [0.1 0.2 0.4 0.3];
theta = [0.6 0.5 0.5 0.3];
phi = [0.5 0.4 0.3 0.2];
%theta = [0.8 0.5];
%phi = [0.7 0.6];
L = 1;
lambda = 1;

for j = 1:N
    % the user context
    %x1 = mvnrnd(userMu,userSigma,1);
    %x2 = mvnrnd(actionMu,actionSigma,1);
    %x = [x1 x2];
    x = (mvnrnd(Mu, Sigma, 1));
    x = x/norm(x);
    contexts(j,:) = x;
    rewardMu = dot(x, theta);
    rewardSigma = get_f_value(phi,x,L);
    reward = mvnrnd(rewardMu,rewardSigma,1);
    rewards(j) = reward;
    
end
for j = 1:N
    if mod(j,500) == 0
        fprintf('Round %d\n', j);
    end
    sContexts = contexts(1:j,:);
    sRewards = rewards(1:j,:);
    
    % estimate theta and phi
    V = (sContexts)' * (sContexts) + lambda * eye(d);
    thetaHat = inv(V) * ((sContexts)') * sRewards;
    varEpsilons = zeros(length(sRewards),1);

    for i = 1: length(sRewards)
        v = sContexts(i,:);
        %varEpsilons(i) = (sRewards(i) - thetaHat' * (v'))^2 - L;
        varEpsilons(i) = get_inverse_f_value((sRewards(i) - thetaHat' * (x'))^2, L);
    end

    phiHat = inv(V) * (sContexts') * varEpsilons;
    thetaHat_history{j} = thetaHat;
    phiHat_history{j} = phiHat;    
    
end

toc;