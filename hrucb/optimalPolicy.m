function armOptimal = optimalPolicy(beta, theta, phi, L, contexts, objective, outcome_type)

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

index = zeros(length(contexts),1);
for i = 1:length(contexts)
    x = contexts{i};
    switch outcome_type
        case "Gaussian"
            normalizedX = (beta - dot(theta,x))/sqrt(get_f_value(phi,x,L));
            if strcmp(objective, "lifetime")
                index(i) = (cdf(pd,normalizedX))^(-1);
            else 
                if strcmp(objective, "reward")            
                    index(i) = dot(theta,x)*(cdf(pd,normalizedX))^(-1);
                end
            end
        case "Student-t"
            student_t_variance = get_f_value(phi,x,L);
            student_t_nu = 2*student_t_variance/(student_t_variance - 1);
            if strcmp(objective, "lifetime")
                index(i) = (tcdf(beta - dot(theta,x),student_t_nu))^(-1);
            else 
                if strcmp(objective, "reward") 
                    index(i) = dot(theta,x) * (tcdf(beta - dot(theta,x),student_t_nu))^(-1);
                end
            end
        otherwise
            normalizedX = (beta - dot(theta,x))/sqrt(get_f_value(phi,x,L));
            if strcmp(objective, "lifetime")
                index(i) = (cdf(pd,normalizedX))^(-1);
            else 
                if strcmp(objective, "reward")            
                    index(i) = dot(theta,x)*(cdf(pd,normalizedX))^(-1);
                end
            end            
    end
end
[M,armOptimal] = max(index);
end

