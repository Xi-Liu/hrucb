function [thetaHat, phiHat, armChosen] = HR_UCB(beta, xi, d, lambda, L, contexts, sContexts, sRewards, objective, outcome_type)

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

phiHat = inv(V) * (sContexts') * varEpsilons;

indexes = zeros(size(contexts,1),1);
for i = 1 : size(contexts,1)
    x = contexts{i};
    u = thetaHat' * (x');
    v = phiHat' * (x');
    switch outcome_type
        case "Gaussian"
            normalizedX = (beta - dot(thetaHat',x))/sqrt(get_f_value(phiHat,x,L));
    
            if strcmp(objective, "lifetime")
                part1 = (cdf(pd,normalizedX))^(-1);
            else 
                if strcmp(objective, "reward") 
                    part1 = u * (cdf(pd,normalizedX))^(-1);
                end
            end
        case "Student-t"
            student_t_variance = get_f_value(phiHat,x,L);
            student_t_nu = 2*student_t_variance/(student_t_variance - 1);
            if strcmp(objective, "lifetime")
                part1 = (tcdf(beta - dot(thetaHat',x),student_t_nu))^(-1);
            else 
                if strcmp(objective, "reward") 
                    part1 = u * (tcdf(beta - dot(thetaHat',x),student_t_nu))^(-1);
                end
            end            
        otherwise
            normalizedX = (beta - dot(thetaHat',x))/sqrt(get_f_value(phiHat,x,L));
    
            if strcmp(objective, "lifetime")
                part1 = (cdf(pd,normalizedX))^(-1);
            else 
                if strcmp(objective, "reward") 
                    part1 = u * (cdf(pd,normalizedX))^(-1);
                end
            end
    end
    part3 = sqrt(x * inv(V) * x');
    indexes(i) = part1 + xi * part3;
end
[M,armChosen] = max(indexes);

end