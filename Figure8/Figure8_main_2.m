%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 8)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% OAMP(BG) + OAMP-MMV-SSL(BG) + OAMP-MMV-ASL(BG)
% ADEP and BER vs SNR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;
tic
addpath('E:\myk\massive access\twc_massive_access_final_1115\');

%% System Parameters
K = 500;     % the number of potential devices
Ka = 50;     % the number of active devices  
T = 10;      % the number of consecutive OFDM symbols
L = 4;    % the modulation order of PSK
M = 70;      % the length of spreading sequence
Sa = Ka/K;
method=char('OAMP(BG)','OAMP-MMV-SSL(BG)','OAMP-MMV-ASL(BG)');

% AMP algorithm paramemters
damp = 0;    % damping value
Iiter = 50;  % the number of iterations  
prior = 0;   % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;  % the threshold of activity detector
var_type = 1;  % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
SNR_dB_set = [0:2:12];              
testall = 3;                % the number of tested algorithms
ADEP = zeros(length(SNR_dB_set), testall);
BER = zeros(length(SNR_dB_set), testall);
NMSE = zeros(length(SNR_dB_set), testall);  

%% Simulation
for alg_sel = [1:testall]
for sim = 1:Nsim
for mm = 1:length(SNR_dB_set) 
    SNR_dB = SNR_dB_set(mm); % specific sample
    
    act_flag = zeros(K,1);
    act_flag(randperm(K,Ka)) = 1; % device activity indicator
    id_act = find(act_flag == 1); % active device set (true)
    
    Xbit = randi([0,L-1], K, T);
    Xdata = pskmod(Xbit, L, 0,'gray');    % transmitted signals
    
    H = ones(K,1); % considering the pre-equalization then channel effect is removed
    
    F = dftmtx(K) / sqrt(K);         
    P = eye(K);
    P = P(randperm(K,M),:);  
    S_wave = P*F;   % spreading code matrix
    
    H_tmp = H(:, ones(1,T));
    act_tmp = act_flag(:, ones(1,T));  
    X = H_tmp.*act_tmp.*Xdata;
    Y = awgn(S_wave*X, SNR_dB, 'measured');    % received signals

    %% JADD
    act_hat = zeros(K,1);
    switch alg_sel
        case 1 % OAMP_Bernoulli_Gaussian
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP(Y, S_wave, damp, Iiter, prior, L);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > 0.3) = 1;      % activity detection
           
        case 2 % OAMP-MMV-SSL_Bernoulli_Gaussian
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP_MMV_SSL(Y, S_wave, damp, Iiter, prior, L);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;

        case 3 % OAMP-MMV-ASL_Bernoulli_Gaussian
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter, prior, L, var_type);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;     
    end
    
    id_act_hat = find(act_hat==1);
    id_inact_hat = find(act_hat==0);
    Xhat(id_inact_hat,:) = 0;
    
    % ADEP
    ADEP(mm,alg_sel) = ADEP(mm,alg_sel) + sum(abs(act_hat-act_flag))/K;

    % NMSE  
    NMSE(mm,alg_sel) = NMSE(mm,alg_sel) + norm(Xhat-X,'fro')^2/norm(X,'fro')^2;
    
    % BER
    id_act_t = intersect(id_act,id_act_hat);           % the set of correctly detected devices
    id_act_f = setdiff(id_act,id_act_t);               % the set of falsely detected devices
    Xhat_demod = pskdemod(Xhat(id_act_t,:), L, 0, 'gray');
    Xhat_demod = reshape(Xhat_demod,[],1);
    Xbit_BER = Xbit(id_act_t,:);
    Xbit_BER = reshape(Xbit_BER,[],1);
    if isempty(id_act_t)
       Err_bit = Ka*T*log2(L);
    else
       Err_bit = sum(abs(de2bi(Xhat_demod,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all') + length(id_act_f)*T*log2(L);
    end   
    BER(mm, alg_sel) = BER(mm, alg_sel) + Err_bit/(Ka*T*log2(L));
    BER_tmp = Err_bit/(Ka*T*log2(L));

    if mod(sim,100) == 0
        fprintf('%s, sim = %d, M = %d, SNR = %d dB,  ADEP = %4.8f, BER = %4.5f, NMSE = %7.9f\n', ...
                method(alg_sel,:), sim, M, SNR_dB_set(mm), ADEP(mm,alg_sel)/sim, BER(mm, alg_sel)/sim, 10*log10(NMSE(mm,alg_sel)/sim));
    end
end
end
toc
end

Perform2.ADEP = ADEP ./ Nsim;
Perform2.BER = BER ./ Nsim;
Perform2.NMSE = 10.*log10(NMSE./Nsim);

%% picture
MarkerSize = 8;
LineWidth = 1.5;
figure;
p2=semilogy(SNR_dB_set,Perform2.ADEP(:,1),'--d','LineWidth',LineWidth,'Color','[0.5 0.1 1] ','MarkerSize',MarkerSize);grid on; hold on
p3=semilogy(SNR_dB_set,Perform2.ADEP(:,2),'--^','LineWidth',LineWidth,'Color','[0 0.4470 0.7410] ','MarkerSize',MarkerSize);grid on; hold on
p4=semilogy(SNR_dB_set,Perform2.ADEP(:,3),'--v','LineWidth',LineWidth,'Color','[0 0.4470 0.7410] ','MarkerSize',MarkerSize);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
gcf_set = [p2,p3,p4];
legend(gcf_set,{'OAMP(BG)','OAMP-MMV-SSL(BG)','OAMP-MMV-ASL(BG)'},'location','southeast','interpreter','latex');

figure;
p2=semilogy(SNR_dB_set,Perform2.BER(:,1),'--d','LineWidth',LineWidth,'Color','[0.5 0.1 1] ','MarkerSize',MarkerSize);grid on; hold on
p3=semilogy(SNR_dB_set,Perform2.BER(:,2),'--^','LineWidth',LineWidth,'Color','[0 0.4470 0.7410] ','MarkerSize',MarkerSize);grid on; hold on
p4=semilogy(SNR_dB_set,Perform2.BER(:,3),'--v','LineWidth',LineWidth,'Color','[0 0.4470 0.7410] ','MarkerSize',MarkerSize);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
gcf_set = [p2,p3,p4];
legend(gcf_set,{'OAMP(BG)','OAMP-MMV-SSL(BG)','OAMP-MMV-ASL(BG)'},'location','southeast','interpreter','latex');
 







