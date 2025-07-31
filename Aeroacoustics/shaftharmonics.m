B = 3; % number of blades
n = 4; % harmonic of the shaft frequency

phase = 0:2*pi/1000:2*pi;
solution = zeros(size(phase));

figure(1)
clf
hold on

for blade = 1:B
    partial_solution = sin(n*phase + (2*pi*n/B)*(blade-1));
    plot(phase,partial_solution,':');
    solution = solution + partial_solution;
end

plot(phase,solution,'k')
shg
