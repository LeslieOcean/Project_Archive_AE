mu=100;
J=1;
m0=2;
L=1;
x=linspace(0,1,100)
thetax=-m0/(2*mu*J)*x.^2+3*m0*L/(mu*J)*x
thetaxrrm=4*m0*L^2/(mu*J*pi^3)*sin(pi*x/L)
plot(x,thetax,'b-',x,thetaxrrm,'g-',LineWidth=2)
xlabel('x')
ylabel('twist angle')
legend('exact','numerical')
