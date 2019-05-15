%%
step = 1000;
xaxis = 1:step:30000;
figure;
plot(xaxis, meanRegret(xaxis), '-gs')