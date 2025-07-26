clear all;
clc;

% Generating the mesh
% Coordinates of nodes
nodes = [0 , 0;
    0 , 2.5e-3;
    0, 5e-3;
    3.33e-3, 5e-3;
    3.33e-3, 2.5e-3;
    3.33e-3, 0;
    6.66e-3, 0;
    6.66e-3, 2.5e-3;
    10e-3, 5e-3;
    10e-3, 0;];
%Connectivity matrix
elements = [2,5,4,3;
    1,6,5,2;
    5,8,9,4;
    7,10,9,8;
    6,7,8,5];

% Material properties
E = 1e9; % Elastic modulus (1 GPa)
nu = 0.3; % Poisson's ratio

% Thickness
thickness = 5e-3; % 5 mm

% Define Gauss quadrature points and weights (for double-point integration)
gauss_points = [-sqrt(1/3), -sqrt(1/3); sqrt(1/3), -sqrt(1/3); sqrt(1/3), sqrt(1/3); -sqrt(1/3), sqrt(1/3)]; % Gauss quadrature points
gauss_weights = [1, 1, 1, 1]; % Gauss quadrature weights


% Initialize global stiffness matrix and force vector
num_nodes = size(nodes, 1);
K = zeros(2 * num_nodes, 2 * num_nodes); % Global stiffness matrix
F = zeros(2 * num_nodes, 1); % Global force vector

% Loop through each element
for elem = 1:size(elements, 1)
    % Extract nodal coordinates for the current element
    elem_nodes = elements(elem, :);
    x = reshape(nodes(elem_nodes, 1), [4,1]);
    y = reshape(nodes(elem_nodes, 2), [4,1]);
    Ke = zeros(8);

    % Define shape functions and their derivatives
    % Define shape functions and their derivatives
    N = [0.25*((1-gauss_points(1,2))*(1-gauss_points(1,1))), 0.25*(1+gauss_points(2,2))*(1-gauss_points(2,1)), 0.25*(1+gauss_points(3,2))*(1+gauss_points(3,2)), 0.25*(1-gauss_points(4,2))*(1+gauss_points(4,1))];
    dNdn = [-0.25*(1-gauss_points(1,2)), 0.25*(1-gauss_points(2,2)), 0.25*(1+gauss_points(3,2)), -0.25*(1+gauss_points(4,2))];
    dNde = [-0.25*(1-gauss_points(1,1)), -0.25*(1+gauss_points(2,1)), 0.25*(1+gauss_points(3,1)), 0.25*(1-gauss_points(4,1))];
   
    dNd1 = dot(x,dNdn');
    dNd2 = dot(y,dNdn');
    dNd3 = dot(x,dNde');
    dNd4 = dot(y,dNde');

    %Jacobian matrix
    J = [dNd3 dNd4; dNd1 dNd2];
    detJ = det(J);
    invJ = inv(J);

    dNdx =  J \[dNdn ; dNde];

    B = zeros(3,8);
    B(1,1:2:7) = dNdx(1, :);
    B(2,2:2:8) = dNdx(2, :);
    B(3,1:2:7) = dNdx(2, :);
    B(3,2:2:8) = dNdx(1, :);

    D = E / (1 - nu^2)*[1, nu, 0; nu, 1, 0; 0, 0, (1-nu)/2];

    Ke = Ke + B' * D * B * detJ*thickness;

    for i = 1:4
        for j = 1:4
            K(2*elem_nodes(i)-1:2*elem_nodes(i),2*elem_nodes(j)-1:2*elem_nodes(j)) = K(2*elem_nodes(i)-1:2*elem_nodes(i),2*elem_nodes(j)-1:2*elem_nodes(j))+ Ke(2*i - 1:2*i, 2*j - 1:2*j);
        end
    end
end


% Apply boundary conditions (imposed displacements)
% Implement boundary conditions here by modifying K and F
K([1,2,3,11,16,17,20], :) = 0;  % Zero out the entire row
K([1,2,3,11,16,17,20], [1,2,3,11,16,18,20]) = 1;  % Set the diagonal element to 1
%Modify the force vector accordingly to displacement BC
for i = 1:2*num_nodes
    if i <= 3 || (i == 11 && i == 16 && i == 17 && i == 20)
        % Node is on the boundary, enforce boundary condition
        F(i) = 0;  % Zero out the entire row
    end
end

%Modify the force vector according to applied forces
F(9)= 42;
F(10)=18;
F(13)= 166.5;
F(14)=500;
F(19)=333;
F
K


% Solve for nodal displacements
% Use appropriate solver (e.g., direct solver or iterative solver)
displacements = K \ F;
displacements

% Find forces at imposed boundary nodes
% Extract forces at boundary nodes based on boundary conditions

% Compute stresses at the four integration points of element A
% For element A, compute stresses (sig_11, sig_22) at integration points
% Implement stress computation based on displacements and element propertie