clear all;

eta = linspace(0, 0.1386, 100);

a = 0.45;
b = 6;

theta2_dimless = a/(b-1)*(exp(-eta)-exp(-b.*eta));

lambda = a/(b-1)*(1-exp((1-b).*eta));

theta2_dimless2 = a/(1-b)*(exp(eta)-exp(b.*eta));

lambda2 = a/(b-1)*(1-exp((b-1).*eta));

t = 0.09*(exp(6.*eta)-exp(eta));
l = -0.09*(exp(5.*eta)-exp(eta));
figure(1);
plot(eta, theta2_dimless2);
hold on;
plot(eta, lambda2);
legend('\theta^2 U_0/\nu L','\lambda');
xlabel('x/L');
ylabel('\theta^2 U_0/\nu L or \lambda')
