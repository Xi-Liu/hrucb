function contexts = generate_context(numActions,contextMu,contextSigma)
    contexts = cell(numActions,1);
    for j = 1:numActions
        x = (mvnrnd(contextMu, contextSigma, 1));
        x = x/norm(x);
        contexts{j} = x;   
    end 
end