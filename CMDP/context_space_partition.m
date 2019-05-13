function centroids = context_space_partition(firstTime, d, numActions, numPartition, numSamples)
if firstTime == 1
    contextMu = zeros(d,1);
    contextSigma = eye(d);
    allContexts = [];
    for i = 1:numSamples
        sampleContext = [];
        for j = 1:numActions
            x = (mvnrnd(contextMu, contextSigma, 1));
            x = x/norm(x);
            sampleContext = [sampleContext, x];
        end
        allContexts = [allContexts; sampleContext];
    end
    % row j is the centroid of cluster j.
    [idx, centroids] = kmeans(allContexts, numPartition, 'MaxIter', 10000);
    save('centroids.mat','centroids');    
else
    load('centroids.mat');
end
end