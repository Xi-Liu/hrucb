function armChosen = contextual_mdp(policy, Nsa, m, numActions)

availableActions = [];

for i = 1:numActions
    if Nsa(1, i) < m
        availableActions = [availableActions, i];
    end
end

if isempty(availableActions)
    armChosen = policy(1);

else
    idx = randi(length(availableActions));
    armChosen = availableActions(idx);
end

end