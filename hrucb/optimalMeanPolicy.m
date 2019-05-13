function armOptimal = optimalMeanPolicy(theta, contexts)

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

index = zeros(length(contexts),1);
for i = 1:length(contexts)
    x = contexts{i};     
    index(i) = dot(theta,x);
end
[M,armOptimal] = max(index);
end

