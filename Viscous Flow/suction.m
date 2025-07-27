eta = linspace(0,5,100);
Pr = [0.5, 1, 2];

f = 1-exp(-eta);
figure(1);
plot(eta, f);
xlabel('\eta');
ylabel('f');

figure(2);
hold on;
for i = 1:3 
    theta = exp(-Pr(i)*eta);
    plot(eta, theta);
end
xlabel('\eta');
ylabel('\theta'); 
legend('Pr=0.5', 'Pr=1', 'Pr=2');
