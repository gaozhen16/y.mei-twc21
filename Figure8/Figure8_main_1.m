%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 8)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% AMP-MMV + OAMP-MMV-SSL + OAMP-MMV-ASL + SWOMP + GSP + SAMP+ Oracle LS +
% Gene-aided OAMP
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
method=char('AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL','SWOMP','GSP','SAMP','Oracle LS','Gene-aided OAMP');

% AMP algorithm paramemters
damp = 0;    % damping value
Iiter = 50;  % the number of iterations  
prior = 1;   % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;  % the threshold of activity detector
var_type = 1;  % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
SNR_dB = [0:2:12];              
testall = 8;                % the number of tested algorithms
ADEP = zeros(length(SNR_dB), testall);
BER = zeros(length(SNR_dB), testall);
NMSE = zeros(length(SNR_dB), testall);  

%% Simulation
for alg_sel = [1:testall]
for sim = 1:Nsim
for mm = 1:length(SNR_dB)   
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
    Y = awgn(S_wave*X, SNR_dB(mm), 'measured');    % received signals

    %% JADD
    act_hat = zeros(K,1);
    switch alg_sel
        case 1 % AMP_MMV
           [Xhat, lambda] = MMV_AMP_Gaussian_EM(Y, S_wave, damp, Iiter, 1e-8, 0); 
           act_hat(mean(lambda,2)> ad_th) = 1;      
       
        case 2 % OAMP-MMV-SSL
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP_MMV_SSL(Y, S_wave, damp, Iiter, prior, L);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;
           
        case 3 % OAMP-MMV-ASL
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter, prior, L, var_type);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;

        case 4  %SWOMP
           epsilon = Sa*10^(-SNR_dB(mm)/10);    
           [Xhat, iter_num] = SW_OMP_Algorithm(Y, S_wave, epsilon); 
           swomp_th = 0.01*max(max(abs(Xhat)));
           act_hat(mean(abs(Xhat),2) > swomp_th) = 1;  
         
        case 5 % GSP
           [Xhat, supp, iter] = DMMV_SP(Y, S_wave, Ka);
           gsp_ad = 0.01*max(max(max(abs(Xhat))));
           gsp_th = 0.9;
           act_hat(sum(sum(abs(Xhat)>gsp_ad,3),2)./T>= gsp_th) = 1; 
           
        case 6 % SAMP
           TH=0.005;
           [Xhat, supp, iter] = SAMP_distributed(Y, S_wave, TH);
           samp_ad = 0.01*max(max(max(abs(Xhat))));
           samp_th = 0.9;
           act_hat(sum(sum(abs(Xhat)>samp_ad,3),2)./T >= samp_th) = 1; 
        
        case 7 % Oracle LS
           S_oracle = S_wave(:,find(act_flag==1));
           Xhat_oracle = (S_oracle' * S_oracle)\(S_oracle'*Y);
           Xhat = zeros(K, T);
           Xhat(find(act_flag==1),:) = Xhat_oracle; 
           act_hat = act_flag;
           
        case 8 % Gene-aided OAMP
           Xhat = zeros(K,T);
           Xhat_tmp = Gene_aided_OAMP(Y, S_wave, damp, Iiter, prior, L, act_flag, SNR_dB(mm), Sa);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat = act_flag;
    end
    
    id_act_hat = find(act_hat==1);
    id_inact = find(act_hat==0);
    Xhat(id_inact,:) = 0;
    
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
                method(alg_sel,:), sim, M, SNR_dB(mm), ADEP(mm,alg_sel)/sim, BER(mm, alg_sel)/sim, 10*log10(NMSE(mm,alg_sel)/sim));
    end
end
end
toc
end

Perform.ADEP = ADEP ./ Nsim;
Perform.BER = BER ./ Nsim;
Perform.NMSE = 10.*log10(NMSE./Nsim);

%% picture
SNR_dB=0:2:12;
MarkerSize = 8;
LineWidth = 1.5;
figure;
p1=semilogy(SNR_dB,Perform.ADEP(:,1),'r-.+','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p5=semilogy(SNR_dB,Perform.ADEP(:,2),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p6=semilogy(SNR_dB,Perform.ADEP(:,3),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p7=semilogy(SNR_dB,Perform.ADEP(:,4),'g-*','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p8=semilogy(SNR_dB,Perform.ADEP(:,5),'k-s','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p9=semilogy(SNR_dB,Perform.ADEP(:,6),'y-o','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on;
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
gcf_set = [p9,p7,p8,p1,p5,p6];
legend(gcf_set,{'SAMP','SWOMP','GSP','AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL'},'location','southwest','interpreter','latex');

figure;
p1=semilogy(SNR_dB,Perform.BER(:,1),'r-.+','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p5=semilogy(SNR_dB,Perform.BER(:,2),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p7=semilogy(SNR_dB,Perform.BER(:,3),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p9=semilogy(SNR_dB,Perform.BER(:,4),'g-*','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p10=semilogy(SNR_dB,Perform.BER(:,5),'k-s','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on;
p11=semilogy(SNR_dB,Perform.BER(:,6),'y-o','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on;
p12=semilogy(SNR_dB,Perform.BER(:,7),'b-','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
p13=semilogy(SNR_dB,Perform.BER(:,8),'-','LineWidth',LineWidth,'Color','[0.4 0.570 0.3410] ','MarkerSize',MarkerSize);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
gcf_set = [p9,p7,p8,p1,p5,p6];
legend(gcf_set,{'SAMP','SWOMP','GSP','AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL'},'location','southwest','interpreter','latex');

 







