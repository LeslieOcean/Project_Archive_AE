clear all;
clc;

n = 5;
node = zeros(10,2);
elem = zeros(5,4);
BC = zeros(10,2);
loading = zeros(10,2);
E = 1000; %1000MPa
v = 0.3;
t = 5;

Ematrix = E/((1+v)*(1-2*v))*[1-v,v,0;v,1,0;0,0,(1-2*v)];

%node list
node(1, :) = [0, 2.5];
node(2, :) = [10/3, 2.5];
node(3, :) = [10/3, 5];
node(4, :) = [0, 5];
node(5, :) = [20/3, 2.5];
node(6, :) = [10, 5];
node(7, :) = [0, 0];
node(8, :) = [10/3, 0];
node(9, :) = [20/3, 0];
node(10, :) = [10, 0];

%element list
elem(1, :) = [1, 2, 3, 4];
elem(2, :) = [2, 5, 6, 3];
elem(3, :) = [7, 8, 2, 1];
elem(4, :) = [8, 9, 5, 2];
elem(5, :) = [9, 10, 6, 5];

%shape functions
syms zeta eta;
N(1) = 1/4*(zeta + 1)*(eta + 1);
N(2) = 1/4*(zeta - 1)*(eta + 1);
N(3) = 1/4*(zeta - 1)*(eta - 1);
N(4) = 1/4*(zeta + 1)*(eta - 1);

%loading list
r = zeros(10,2);
r(6,1) = 42;
r(10,1) = 18;
r(4,2) = 333.3/2;
r(3,2) = 500;
r(6,2) = 333.3;
r = reshape(r',[20,1]);

%BC list
d = ones(10,2);
b = 0;
[d(1,1), d(4,1), d(7,1), d(7,2), d(8,2), d(9,2), d(10,2)] = deal(b,b,b,b,b,b,b);
d = reshape(d', [20,1]);

%stiffness matrix
for i = 1:5
    coordinate = node(elem(i,:),:);
    syms x y;
    x = N * coordinate(:,1);
    y = N * coordinate(:,2);
    
    J(:,:,i) = [diff(x,zeta), diff(y,zeta); diff(x,eta), diff(y,eta)];
    J_1 = inv(J(:,:,i));
    B1 = J_1*[diff(N,zeta); diff(N,eta)];
    
    B(1,1:2:7,i) = B1(1,:);
    B(2,2:2:8,i) = B1(2,:);
    B(3,1:2:7,i) = B1(1,:);
    B(3,2:2:8,i) = B1(2,:);

    k(:,:,i) = localstiff(i, B(:,:,i), Ematrix, J(:,:,i), zeta, eta);
end

K = globalstiff(k, elem);
%K = K_Global;
K1 = K;
for h = 1:20
    if d(h) == 0
        K(h,:) = 0;
        K(h, h) = 1;
    end
end


%nodal displacement
displacement = K \ r;
D = reshape((K \ r), [2,10]);
disp(D');

%reaction force
R = K1 * displacement;
disp(reshape(R, [2,10])');

%stress for element A
r1 = reshape(D(:,elem(5,:)),[8,1]);
disp(r1);
strain = B(:,:,5)* r1;
stress = Ematrix*strain;
a = sqrt(3)/3;
stress_gauss = double([subs(stress, [zeta,eta],[a,a]);subs(stress, [zeta,eta],[-a,a]);subs(stress, [zeta,eta],[-a,a]);subs(stress, [zeta,eta],[-a,-a])]);
disp(stress_gauss);

function k1 = localstiff(i, B, E, J, zeta, eta)
    a = sqrt(3)/3;
    f = B'*E*B*det(J);
    k1 = subs(f, [zeta, eta],[a, a]) + subs(f, [zeta, eta],[-a, a]) + subs(f, [zeta, eta],[a, -a]) + subs(f, [zeta, eta],[-a, -a]);
end

function K = globalstiff(k, elem)
K_Global = zeros(20,20);
for ii = 1:5
    A = [elem(ii,:)];
    I_exact = k(:,:,ii);
    for m = 1:8
        B = [2*A(1,1)-1, 2*A(1,1), 2*A(1,2)-1, 2*A(1,2), 2*A(1,3)-1, 2*A(1,3), 2*A(1,4)-1, 2*A(1,4)];
        if rem(m,2) == 1
            pos = A(1, (m-1)/2 + 1);
            for l = 1: 8
                K_Global(B(l), 2*pos-1) = K_Global(B(l), 2*pos-1) + I_exact(l, m);
            end
        elseif rem(m,2) == 0
            pos = A(1, m/2);
            for h = 1:8
                K_Global(B(h), 2*pos) = K_Global(B(h), 2*pos) + I_exact(h, m);
            end
        end
    end
end
K = K_Global;
end
