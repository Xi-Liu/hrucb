function [thetaHat, armChosen] = HR_UCB_fixed_sigma(beta, xi, d, lambda, L, contexts, sContexts, sRewards, sigma_max, objective, outcome_type)

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

V = (sContexts)' * (sContexts) + lambda * eye(d);

thetaHat = inv(V) * ((sContexts)') * sRewards;


varEpsilons = zeros(length(sRewards),1);

for i = 1: length(sRewards)
    x = sContexts(i,:);
    varEpsilons(i) = get_inverse_f_value((sRewards(i) - thetaHat' * (x'))^2, L);
end

%phiHat = inv(V) * (sContexts') * varEpsilons;

indexes = zeros(size(contexts,1),1);
for i = 1 : size(contexts,1)
    x = contexts{i};
    u = thetaHat' * (x');
    %v = phiHat' * (x');
    normalizedX = (beta - dot(thetaHat',x))/sigma_max;
    if strcmp(objective, "lifetime")
        part1 = (cdf(pd,normalizedX))^(-1);
    else
        if strcmp(objective, "reward")
            part1 = u * (cdf(pd,normalizedX))^(-1);
        end
    end
    part3 = sqrt(x * inv(V) * x');
    indexes(i) = part1 + xi * part3;
end
[M,armChosen] = max(indexes);

end