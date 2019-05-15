function [thetaHat, armChosen] = LinUCB(contexts, A, b, alpha)
    % Estimate theta
    A_inv = inv(A);
    thetaHat = A_inv * b;
    index = zeros(size(contexts,1),1);
    
    for i=1:size(contexts,1)
        index(i) = contexts{i}*thetaHat + alpha*sqrt(contexts{i}*A_inv*((contexts{i})'));
    end
    [M,armChosen] = max(index);
end