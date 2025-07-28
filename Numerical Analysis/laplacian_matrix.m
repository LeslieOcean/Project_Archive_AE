clc;

hx = 2;
hy = 2;

Dx = 1/hx * [-1 1 0; 0 -1 1];
Dy = 1/hy * [-1 1 0; 0 -1 1];

Lxx = Dx.'*Dx;
Lyy = Dy.'*Dy;

Ix = eye(3);
Iy = eye(3);

A = kron(Iy,Lxx) + kron(Lyy,Ix);

Du=0.05;
Dv=1;
kappa=5;
I = eye(9);

J011 = Du*A+kappa*I;
J022 = Dv*A;

figure;
hold on;
grid on;
axis equal;
xlabel('Re(z)');
ylabel('Im(z)');
for i = 1:9
    r=0;
    for j = 1:9
        if i ~= j
            r = r+abs(J022(i,j));
        end
    end
    disp(r);
    x = J022(i,i);
    y = 0;
    plot(x,y, 'ro', 'MarkerFaceColor','r');
    fplot(@(t) x + r*cos(t), @(t) y + r*sin(t), [0, 2*pi]);
end

hold off;
