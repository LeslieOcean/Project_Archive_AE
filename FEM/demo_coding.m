% Initializing the environment
clear all;
clc;

% Node and coordinate definitions
Nodes = 1:10;
Coordinates = [
    0,0; 3.33,0; 6.66,0; 10,0; 0,2.5; 
    3.33,2.5; 6.66,2.5; 0,5; 3.33,5; 10,5
];
Elements = [
    6, 5, 1, 2; 7, 6, 2, 3; 10, 7, 3, 4; 
    9, 8, 5, 6; 10, 9, 6, 7
];

% Shape function construction
syms zeta eta;
N(1) = (1+zeta)*(1+eta)/4;
N(2) = (1-zeta)*(1+eta)/4;
N(3) = (1-zeta)*(1-eta)/4;
N(4) = (1+zeta)*(1-eta)/4;

% Iterating through elements
for jj = 1:5
    % Geometric transformation
    elemCoordinates = Coordinates(Elements(jj,:)',:);
    syms x y;
    x = N * elemCoordinates(:,1);
    y = N * elemCoordinates(:,2);
    
    J(:,:,jj) = [diff(x,zeta), diff(y,zeta); diff(x,eta), diff(y,eta)];

    B1 = inv(J(:,:,jj)) * [diff(N,zeta); diff(N,eta)];
    
    % Creating B matrix
    trial = [B1(1,:); zeros(1,numel(N))];
    B(1,:,jj) = reshape(trial,[1,8]);
    
    trial = [zeros(1,numel(N)); B1(2,:)];
    B(2,:,jj) = reshape(trial,[1,8]);
    
    trial = [B1(2,:); B1(1,:)];
    B(3,:,jj) = reshape(trial,[1,8]);
end

shapeDiff = B;

% Applying plane strain condition
E = 1e3;    % Young's Modulus
g = 0.3;    % Poisson ratio
E_matrix = E / ((1+g) * (1-2*g)) * [1-g, g, 0; g, 1-g, 0; 0, 0, 0.5*(1-2*g)];

% Gauss Quadrature Integration
k = zeros(8,8,5);
for kk = 1:5
    I(:,:,kk) = B(:,:,kk)' * E_matrix * B(:,:,kk) * det(J(:,:,kk));
    a = 1/sqrt(3);
    k(:,:,kk) = double(k(:,:,kk) + subs(I(:,:,kk), [zeta, eta], [a,a]) + subs(I(:,:,kk), [zeta, eta], [a,-a]) + subs(I(:,:,kk), [zeta, eta], [-a,-a]) + subs(I(:,:,kk), [zeta, eta], [-a,a]));
    sanityCheck(kk) = issymmetric(k(:,:,kk));
end

% Expanding local to global size
K_Global = zeros(20,20);
for ii = 1:5
    A = [Elements(ii,:)];
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

globalStiffness = K_Global;
R = zeros(10,2);
R(8,2) = 33.33/2;
R(9,2) = (33.33 + 66.66)/2;
R(10,2) = 66.66/2;
R(10,1) = 4.2;
R(4,1) = 1.8;
R = R * 1e3;
R = reshape(R', [20,1]);
u = ones(1,10);
v = ones(1,10);
v(1:4) = 0;
u(1) = 0;
u(7) = 0;
u(8) = 0;
d = [u; v];
D = reshape(d, [20,1]);

% Applying boundary conditions
for kk = 1:numel(D)
    if (D(kk) == 0)
        disp(['Zero displacement applied at index: ', num2str(kk)]);
        globalStiffness(kk,:) = 0;
        globalStiffness(kk,kk) = 1;
    end
end

% Solving for nodal displacements
nodalDisplacements = reshape(linsolve(globalStiffness, R), [2,10])

% Stress calculation using plane strain condition for Element 3
nodesA = Elements(3,:);
loadVecA = reshape(nodalDisplacements(:,nodesA), [8,1]);
strain = shapeDiff(:,:,3) * loadVecA;
stress = E_matrix * strain;

% Finding stress values at Gauss points
a = 1/sqrt(3);
S_GP(:,1) = subs(stress, [zeta, eta], [a, a]);
S_GP(:,2) = subs(stress, [zeta, eta], [-a, a]);
S_GP(:,3) = subs(stress, [zeta, eta], [-a, -a]);
S_GP(:,4) = subs(stress, [zeta, eta], [a, -a]);

SGP = double(S_GP)


