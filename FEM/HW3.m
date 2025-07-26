clc;

%basic input
N = 15; % the number of elements
a = 3; % the length of plate is 3m
b = 2; % the width of plate is 2m
Q = 0; % the distributed heat source
kxx = 45; % thermal conductivities in x direction
kyy = 45; % thermal conductivities in y direction
Tem1 = 200 + 273.15; % temperature at boundary 1 in Kelvin
Tem2 = 25 + 273.15; % temperature at boundary 2 in Kelvin

%initializations
n = 14; % number of nodes

T = zeros(n, 1);
q = zeros(n, 1); %temperary load vector
node = zeros(14, 2); %nodes location
elem = zeros(15, 3);  %elements and their vertices
[node(1, 1), node(2, 1), node(3, 1), node(1, 2), node(4, 2), node(8, 2), node(11, 2)] = deal(0, 0, 0, 0, 0, 0, 0);
[node(2, 2), node(9, 1), node(4, 1), node(5, 1), node(6, 1), node(7, 1), node(9, 2)] = deal(1, 1, 1, 1, 1, 1, 1);
[node(5, 2), node(12, 2)] = deal(0.5, 0.5);
[node(6, 2), node(13, 2)] = deal(1.5, 1.5);
[node(8, 1), node(9, 1), node(10, 1), node(3, 2), node(7, 2), node(10, 2), node(14, 2)] = deal(2, 2, 2, 2, 2, 2, 2);
[node(11, 1), node(12, 1), node(13, 1), node(14, 1)] = deal(3, 3, 3, 3);
elem(1, :) = [1, 2, 5];
elem(2, :) = [2, 3, 6];
elem(3, :) = [5, 2, 6];
elem(4, :) = [4, 1, 5];
elem(5, :) = [6, 3, 7];
elem(6, :) = [6, 7, 10];
elem(7, :) = [9, 6, 10];
elem(8, :) = [5, 6, 9];
elem(9, :) = [8, 5, 9];
elem(10, :) = [8, 4, 5];
elem(11, :) = [11, 8, 12];
elem(12, :) = [8, 9, 12];
elem(13, :) = [12, 9, 13];
elem(14, :) = [9, 10, 13];
elem(15, :) = [13, 10, 14];
[T(1, 1), T(2, 1), T(3, 1)] = deal(Tem1, Tem1, Tem1);
[T(11, 1), T(12, 1), T(13, 1), T(14, 1)] = deal(Tem2, Tem2, Tem2, Tem2);


kk = localstiffness(6, node, elem); %local stiffness matirx
disp(kk);
KK = assembly(node, elem); %global stiffness matrix
%disp(KK);
Kff = KK(4:10, 4:10); %using partitioning K and q
%disp(Kff);
qu = q(4:10, :);
C1 = KK(4:10, 1:3)*T(1:3, :);
C2 = KK(4:10, 11:14)*T(11:14, :);
Rf = qu - C1 - C2;
T(4:10, :) = linsolve(Kff, Rf);
disp(T); %output sees below
%                    473.15
%                    473.15
%                    473.15
%         414.816666666667
%         414.816666666666
%         414.816666666666
%         414.816666666667
%         356.483333333333
%         356.483333333333
%         356.483333333333
%                    298.15
%                    298.15
%                    298.15
%                    298.15
q(1:3, :) = KK(1:3, :)*T;
q1 = q(1, :);
q(11:14, :) = KK(11:14, :)*T;
q2 = q(14, :)*2;
%patch('Vertices', node, 'Faces', elem, 'EdgeColor','green','facecolor','none')

function k1 = localstiffness(m, node, elem) %local stiffness matrix k function
kappa = 45;
k = zeros(3, 3);
area = 0.5*abs(det([node(elem(m, 1), :)-node(elem(m, 2), :); node(elem(m, 1), :)-node(elem(m, 3), :)]));
b1 = node(elem(m, 2), 2) - node(elem(m, 3), 2);
b2 = node(elem(m, 3), 2) - node(elem(m, 1), 2);
b3 = node(elem(m, 1), 2) - node(elem(m, 2), 2);
c1 = node(elem(m, 3), 1) - node(elem(m, 2), 1);
c2 = node(elem(m, 1), 1) - node(elem(m, 3), 1);
c3 = node(elem(m, 2), 1) - node(elem(m, 1), 1);
k(1, 1) = b1^2 + c1^2;
k(1, 2) = b1*b2 + c1*c2;
k(1, 3) = b1*b3 + c1*c3;
k(2, 2) = b2^2 + c2^2;
k(2, 3) = b2*b3 + c2*c3;
k(3, 3) = b3^2 + c3^2;
k(2, 1) = k(1, 2);
k(3, 1) = k(1, 3);
k(3, 2) = k(2, 3);
k1 = k.*(kappa/4/area);
end

function K = assembly(node, elem) % global stiffness matrix K function
n = 14;
N = 15;
K = zeros(n, n);
for m = 1:N
    Kt = localstiffness(m, node, elem);
    for i = 1:3
        for j = 1:3
            K(elem(m, i), elem(m, j)) =  K(elem(m, i), elem(m, j)) + Kt(i ,j);
        end
    end
end
end

        