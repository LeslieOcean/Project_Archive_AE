
%% Signal
figure;
dataA = MIC{1}.pMic{2};
dt = 1/fS; %sample time [s]
Naq = length(dataA);    % # of acquisition samples
plot(0:dt:dt*(Naq-1),dataA); hold on;
legend('AoA 5, AoS 0, J=1.6'); axis([0 6 -30 30]);
xlabel('t (s)'); ylabel('p(t) (Pa)');
title('Time series of acoustic pressure in Pa');



%% Spectral Analysis
NB = 6;             %number of blade
N = 2^13;           %width of window
B = floor(Naq/N);   % # of enemble averages per data file (Nfile samples), [-]
df = fS/N;          % frequency resolution, [Hz]
flab = (0:N-1)*df;  % frequency discretization, [Hz]
T = dt*N;           % recording time one partition, [s]
windowing = true;   % hannding window or not? (not much different for broadband signal)
p_ref = 20e-6;     % 20 micro-Pa reference pressure, [Pa]

x_m = 0.55;
y_m = 0.44;
z_m = 0.43;
r1 = sqrt(x_m^2+y_m^2+z_m^2);  % microphone position w.r.t. starboard propeller [m]
n = 123.03; % RPM [Hz]
BPF = n*NB;

%% prop_off vs. prop_on
p_raw_1 = MIC{1}.pMic{1};
p_raw_2 = MIC{1}.pMic{2};
p_raw_off = MIC{1}.pMic{3};
p_3 = MIC{1}.pMic{13};

% compute OASPL
OASPL_1 = 20*log10(std(p_raw_1)/p_ref);
OASPL_2 = 20*log10(std(p_raw_2)/p_ref);
OASPL_off = 20*log10(std(p_raw_off)/p_ref);

[~,flab_1,~,df,phi_1,N] = fcn_spectrumN_V1(N,1/fS,p_raw_1,2); % [bst = 1: spatial-data, bst = 2: time-data]
[~,flab_2,~,df,phi_2,N] = fcn_spectrumN_V1(N,1/fS,p_raw_2,2);
[~,flab_off,~,df,phi_off,N] = fcn_spectrumN_V1(N,1/fS,p_raw_off,2);
[~,flab_off,~,df,phi_3,N] = fcn_spectrumN_V1(N,1/fS,p_3,2);

% Parseval's theorem: check scaling-----------------------------------
% signal's energy - integrating the spectrum:
p2int_1 = trapz(phi_1(1:N/2,1))*df; % Trapezoidal-integration: Int[PSD function(f)]df, [unit^2 = energy], slightly less accurate: p2int = sum(Guu(1:N/2,1))*df;
% signal's energy - same as variance? ratio = 1?
display(strcat(['Ratio = ',num2str(p2int_1/(std(p_raw_1)).^2)])); % CHECK?
% --------------------------------------------------------------------

% compute SPSL
SPSL_1 = 20*log10(sqrt(phi_1/p_ref^2));
SPSL_2 = 20*log10(sqrt(phi_2/p_ref^2));
SPSL_off = 20*log10(sqrt(phi_off/p_ref^2));
SPSL_3 = 20*log10(sqrt(phi_3/p_ref^2));

% correction
SPSL_cor_on_1 = convectionfcn(opp{1}.vInf(1), opp{1}.temp(1), x_m, r1, y_m, z_m, SPSL_1);
SPSL_cor_on_2 = convectionfcn(opp{1}.vInf(22), opp{1}.temp(22), x_m, r1, y_m, z_m, SPSL_2);
SPSL_cor_off = convectionfcn(opp{2}.vInf(1), opp{2}.temp(1), x_m, r1, y_m, z_m, SPSL_off);
SPSL_cor_3 = convectionfcn(opp{2}.vInf(1), opp{2}.temp(1), x_m, r1, y_m, z_m, SPSL_3);

% scaling effect for frequency
D_ATR = 3.93;   % full model propeller diameter;
U_cruise = 143; % maximum cruise speed [m/s];
flab_scaled = flab_1*(D/D_ATR)*(U_cruise/opp{1}.vInf(1));

figure;
plot(flab_scaled(1:N/2)/BPF,SPSL_cor_on_1(1:N/2),'green');
hold on;
plot(flab_scaled(1:N/2)/BPF,SPSL_cor_on_2(1:N/2),'blue');
hold on;
plot(flab_scaled(1:N/2)/BPF,SPSL_cor_off(1:N/2),'red');
% hold on;
% plot(flab_scaled(1:N/2)/BPF,SPSL_cor_3(1:N/2),'black');
legend('AoA=0 deg','AoA=5 deg','AoA=8 deg');
axis([0 6 -10 100]); xlabel('f/BPF[-]'); ylabel('SPSL[dB/Hz]');
grid on;
xticks(0:1:6);
% 
% 
% % spectrum in dB/Hz (so basically log-log)
% figure;
% semilogx(flab(1:N/2),SPSL(1:N/2),"red");
% hold on;
% semilogx(flab(1:N/2),SPSL_cor(1:N/2),"blue");
% hold on;
% semilogx(flab_scaled(1:N/2), SPSL_cor(1:N/2),"green");
% grid on;
% legend('Raw', 'Remove Convection Effect', 'Full-scale Model','Location', 'southeast');
% axis([1 100000 -10 100]); xlabel('f[Hz]'); ylabel('SPSL[dB/Hz]');
% %title('Acoustic spectrum in dB/Hz, with log-axis of frequency');
% 
% %% scaled with BPF
% figure;
% plot(flab_scaled(1:N/2)/BPF, SPSL_cor(1:N/2))
% 
% 
% fprintf('OASPL (raw): %.2f dB\n', OASPL);
%     end
% end
function SPSL_cor = convectionfcn(Uinf, T, x_m, r1, y_m, z_m, SPSL)
    M = Uinf/sqrt(1.4*287.05*T);
    xi = sqrt((1-M*x_m/r1)^2 - (x_m^2+y_m^2)/r1^2);
    pc_pm= 1/2*(xi*r1/z_m+(1-M*x_m/r1)^2)*sqrt(1+M^2*(xi^2+y_m^2/r1^2)); %ratio between corrected pressure and measured pressure
    rm_rc = (1-M*x_m/r1)/sqrt(1+M^2*(xi^2+y_m^2/r1^2));
    SPSL_cor = SPSL + 20*log10(pc_pm) + 20*log10(rm_rc);
end







