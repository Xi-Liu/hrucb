%%
step = 1000;
xaxis = 1:step:30000;
norm_phiHat_minus_phi = zeros(length(xaxis),1);
for i=1:length(xaxis)
    norm_phiHat_minus_phi(i) = norm(phiHat_history{1+(i-1)*step} - phi');
end
figure;
plot(xaxis, norm_phiHat_minus_phi, '-rs')