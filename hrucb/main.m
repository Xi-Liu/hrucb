clear all;
clc;
tic;

% load a configuration file
config.config_20arms_icml_discrete

% for reproducibility
rng default  


% generate rewards on the fly
results = cell(total_repeat, 1);

thetaHat_history = cell(T, total_repeat);
phiHat_history = cell(T, total_repeat);
action_history = zeros(T, total_repeat);
optimal_action_history = zeros(T, total_repeat);
optimal_mean_action_history = zeros(T, total_repeat);

for repeat = 1 : total_repeat
    % Initialize the stored contexts and rewards
    sContexts = zeros(ceil(T*K),d);
    sRewards = zeros(ceil(T*K),1);
    sContexts_count = 0;
    
    % For LinUCB
    A = zeros(d,d);
    b = zeros(d,1);
    
    for i = 1:initialRounds
        beta = generate_beta();
        contexts = cell(numActions,1);
        
        % User-action contexts
        switch context_gen_mode
            case "randomly-generate-new-contexts" 
                contexts = generate_context();
            case "read-contexts-from-file"
                context.context_10_discrete_icml              
            otherwise
                fprintf("No match for mode of context generation!!\n");               
        end
    
        rewardMus = cell(numActions,1);
        rewardSigmas = cell(numActions,1);
    
        for j = 1:numActions
            rewardMus{j} = dot(contexts{j}, theta);
            rewardSigmas{j} = dot(contexts{j},phi)+L;
        end
    
        live = true;
        while (live == true)
            armChosen = randi(numActions);
            A = A + ((contexts{armChosen})')*(contexts{armChosen});
            observedReward = get_outcome(rewardMus{armChosen},rewardSigmas{armChosen},1, outcome_type);
            b = b + observedReward*((contexts{armChosen})');
            if sContexts_count < K*i
                sContexts(sContexts_count+1,:) = contexts{armChosen};
                sRewards(sContexts_count+1) = observedReward;
                sContexts_count = sContexts_count + 1;
            end
            if observedReward < beta
                live = false;
            else
                live = true;
            end
        end
    end
    accumuRegrets = zeros(T,1);
    
    for i = 1:T
        if mod(i,500) == 0
            fprintf('User %d\n', i);
        end
        phiHat = zeros(d,1);
        % already perform some rounds before it really starts
        % t = initialRounds - 1 + i;
        
        t = length(sRewards);

        alpha1 = sigmaMax2 * sqrt(d * log((t+lambda)/(lambda * delta)))+sqrt(lambda);
        
        % gradually reduce the vlaue of delta
        deltaTemp = delta / (t^2);

        rho = Mf * alpha1 * deltaTemp / 3 * (1 + 2 * alpha3 * deltaTemp / 3) + sqrt(lambda) + alpha2 * deltaTemp / 3;

        xi = C3 * alpha1 + C4 * rho * deltaTemp;

        beta = generate_beta();
        contexts = cell(numActions,1);
        % the user context
        switch context_gen_mode
            case "randomly-generate-new-contexts" 
                contexts = generate_context();
            case "read-contexts-from-file"
                context.context_10_discrete_icml
            otherwise
                fprintf("No match for mode of context generation!!\n");               
        end

        rewardMus = cell(numActions,1);
        rewardSigmas = cell(numActions,1);

        for j = 1:numActions
            rewardMus{j} = dot(contexts{j}, theta);
            rewardSigmas{j} = dot(contexts{j},phi)+L;
        end
        live = true;

        % the optimal policy
        armOptimal = optimalPolicy(beta, theta, phi, L, contexts, objective, outcome_type);
        armMeanOptimal = optimalMeanPolicy(theta, contexts);
        optimal_action_history(i, repeat) = armOptimal;
        optimal_mean_action_history(i, repeat) = armMeanOptimal;
        % the HR_UCB policy
        switch alg
            case 'HR-UCB'
                [thetaHat, phiHat, armChosen] = HR_UCB(beta, xi, d, lambda, L, contexts, sContexts(1:sContexts_count,:), sRewards(1:sContexts_count,1), objective, outcome_type);
            case 'LinUCB'
                [thetaHat, armChosen] = LinUCB(contexts, A, b, alpha);
            case 'HR-UCB-fixed-sigma'
                [thetaHat, armChosen] = HR_UCB_fixed_sigma(beta, xi, d, lambda, L, contexts, sContexts(1:sContexts_count,:), sRewards(1:sContexts_count,1), sigmaMax2);
            otherwise
                [thetaHat, phiHat, armChosen] = HR_UCB(beta, xi, d, lambda, L, contexts, sContexts(1:sContexts_count,:), sRewards(1:sContexts_count,1));
        end
        action_history(i, repeat) = armChosen;
        
        %accumObservedRewards = 0;
        while (live == true)
            A = A + ((contexts{armChosen})')*(contexts{armChosen});
            observedReward = get_outcome(rewardMus{armChosen},rewardSigmas{armChosen},1, outcome_type);
            b = b + observedReward*((contexts{armChosen})');
            if sContexts_count < K*(i+initialRounds)
                sContexts(sContexts_count+1,:)= contexts{armChosen};
                sRewards(sContexts_count+1,:) = observedReward;
                sContexts_count = sContexts_count + 1;
            end
            %accumObservedRewards = accumObservedRewards + observedReward;
            if observedReward < beta
                live = false;
            else
                live = true;
            end
        end
        
        thetaHat_history{i,repeat} = thetaHat;
        phiHat_history{i,repeat} = phiHat;
        mu = 0;
        sigma = 1;
        pd = makedist('Normal','mu',mu,'sigma',sigma);
        x_opt = contexts{armOptimal}; 
        x_emp = contexts{armChosen};
        switch outcome_type
            case "Gaussian"
                normalizedX = (beta - dot(theta,x_opt))/sqrt(get_f_value(phi,x_opt,L));
                if strcmp(objective , "lifetime")
                    accumOptimalRewards = (cdf(pd,normalizedX))^(-1);
                else
                    if strcmp(objective, "reward")
                        accumOptimalRewards = dot(theta,x_opt)*(cdf(pd,normalizedX))^(-1);
                    else
                    fprintf("Specify the objective!\n")
                    end
                end
                  
                normalizedX_emp = (beta - dot(theta,x_emp))/sqrt(get_f_value(phi,x_emp,L));
                if strcmp(objective , "lifetime")
                    accumRewards = (cdf(pd,normalizedX_emp))^(-1);
                else
                    if strcmp(objective, "reward")
                        accumRewards = dot(theta,x_emp)*(cdf(pd,normalizedX_emp))^(-1);
                    else
                    fprintf("Specify the objective!\n")
                    end
                end
            case "Student-t"    
                student_t_variance = get_f_value(phi,x_opt,L);
                student_t_nu = 2*student_t_variance/(student_t_variance - 1);
                if strcmp(objective, "lifetime")
                    accumOptimalRewards = (tcdf(beta - dot(theta,x_opt),student_t_nu))^(-1);
                else 
                    if strcmp(objective, "reward") 
                        accumOptimalRewards = dot(theta,x_opt) * (tcdf(beta - dot(theta,x_opt),student_t_nu))^(-1);
                    end
                end
                
                student_t_variance_emp = get_f_value(phi,x_emp,L);
                student_t_nu_emp = 2*student_t_variance_emp/(student_t_variance_emp - 1);
                if strcmp(objective, "lifetime")
                    accumRewards = (tcdf(beta - dot(theta,x_emp),student_t_nu_emp))^(-1);
                else 
                    if strcmp(objective, "reward") 
                        accumRewards = dot(theta,x_emp) * (tcdf(beta - dot(theta,x_emp),student_t_nu_emp))^(-1);
                    end
                end                
                
        end
        individualRegret = accumOptimalRewards - accumRewards;
        accumuRegrets(i) = individualRegret;
    end
    results{repeat} = accumuRegrets;
end

meanRegret = 0;
for i = 1:total_repeat
    meanRegret = meanRegret + cumsum(results{i});
end
meanRegret = meanRegret/total_repeat;
save(strcat(alg, '_', objective, '_', context_gen_mode, '_', outcome_type, '_K=', num2str(K), '_beta=-2,0_', 'T=', num2str(T), '_repeat=', num2str(total_repeat), '.mat'))

toc;