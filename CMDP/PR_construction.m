function [P, R] = PR_construction(numStates, numActions, m, Nssa, Nsa, Rsa)
% P is numStates by numStates by numActions
% R is numStates by numActions
P = zeros(numStates, numStates, numActions);
R = zeros(numStates, numActions);
for i = 1:numActions
    P(2,2,i) = 1; 
    % P(2,1,i) = 0; % since it is already 0 thus ignored
    % R(2, i) = 0; % since it is already 0 thus ignored
    
    if Nsa(1,i) < m
        R(1,i) = 1;
    else
        R(1,i) = Rsa(1,i)/Nsa(1,i);
    end
    
    
    for j = 1:numStates
        if Nsa(1,i) < m
            if j == 1
                P(1,j,i) = 1;
            else
                P(1,j,i) = 0;
            end
        else
            P(1,j,i) = Nssa(1,j,i)/Nsa(1,i);
        end
    end
    
end



end