clear all;

a = pi/2;
eta = linspace(0,1,100);

A = cosh(a.*eta).*sin(a.*eta)/cosh(a);
B = sinh(a.*eta).*cos(a.*eta)/cosh(a);
abs_amp = sqrt(A.^2+B.^2);

figure(1);
plot(eta, abs_amp);
ylabel('Absolute Amplitude [-]');
xlabel('\eta [-]');

figure(2);
plot(eta, A);
hold on;
plot(eta, B);
ylabel('Amplitude Function [-]');
xlabel('\eta [-]');
legend('A(\eta)', 'B(\eta)');

%%
figure(3);
omegat = [0, pi/4, pi/2, pi*3/4, pi];
for i = 1:5
    u_ = A.*cos(omegat(i))+B.*sin(omegat(i));
    plot(eta, u_);
    hold on;
end
xlabel('\eta [-]');
ylabel('u/U_0 [-]');
legend('\omega t = 0', '\omega t = \pi/4', '\omega t = \pi/2','\omega t = 3\pi/4', '\omega t = \pi', 'Location', 'best');


%%
falkner_skan = @(eta, y) [y(2);                           % y1' = y2 (f')
                           y(3);                           % y2' = y3 (f'')
                           -y(1)*y(3) - y(2)^2 + 1];       % y3' = -f f'' - (f')^2 + 1

eta_span = [0 10];      
fw = 2.25;              
fw11 = 1.75;            
fw22 = -1.75;          

y0_case1 = [fw; 0; fw11];   
y0_case2 = [fw; 0; fw22];   

[eta1, y1] = ode45(falkner_skan, eta_span, y0_case1);
[eta2, y2] = ode45(falkner_skan, eta_span, y0_case2);

figure;
plot(y1(:,2),eta1, 'b-', 'LineWidth', 1.5); hold on;
plot(y2(:,2),eta2, 'r--', 'LineWidth', 1.5);
ylabel('\eta');
xlabel('f''(\eta)');
legend('f''''(0) = 1.75', 'f''''(0) = -1.75', 'Location', 'best');
grid on;