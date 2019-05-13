function output = get_f_value(phi, x, L)
    output = max(0.01, (dot(phi,x)+L));
    %output = exp(dot(phi,x));
end