
data = importdata('velocity.txt');

y = data.data(:,2)*0.001;
uUe = data.data(:,3);

nu = 15.0*10^(-6);
Ue = 9.804;
kappa = 0.41;
B = 5;
Rey = y.*Ue/nu;

logLaw = @(a, Rey) (a/kappa) .* log(Rey * a) + B * a;
a0 = 0.039;

Cf = 2 * a0^2;

%uUe_theo = (a/kappa).*log(Rey.*a)+B*a;
uUe_theo = logLaw(a0, Rey);

figure(1);
semilogx(Rey, uUe,'b',Rey, uUe_theo, 'r');
%hold on;
%plot(Rey, uUe_theo, color='r');
xlabel('Re_y');
ylabel('u/U_e');
legend('Measured Data','Fit Curve C_f=0.003')
grid();

