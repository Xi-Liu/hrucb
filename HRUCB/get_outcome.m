function outcome = get_outcome(mu, sigma, n, type)
% mu: mean of the random variable
% sigma: variance of the random variable
% n: number of samples
% nu = (2*\sigma^2)/(\sigma^2-1)
    switch type
        case "Gaussian"
            outcome = mvnrnd(mu,sigma,n);
        case "Student-t"
            nu = (2*(sigma^2))/((sigma^2)-1);
            outcome = mu + trnd(nu,n);
        otherwise
            outcome = mvnrnd(mu,sigma,n);
    end
end