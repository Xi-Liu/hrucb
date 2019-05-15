function armOptimal = optimalPolicy(beta, theta, phi, L, contexts)

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

index = zeros(length(contexts),1);
for i = 1:length(contexts)
    x = contexts{i};
    normalizedX = (beta - dot(theta,x))/sqrt(get_f_value(phi,x,L));
    index(i) = (cdf(pd,normalizedX))^(-1);
end
[M,armOptimal] = max(index);
end

