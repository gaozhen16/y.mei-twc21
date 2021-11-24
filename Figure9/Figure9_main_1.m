%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 9)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% AMP-MMV + OAMP-MMV-SSL + OAMP-MMV-ASL + SWOMP + GSP + SAMP+ Oracle LS +
% Gene-aided OAMP
% ADEP and BER vs T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; 
tic
addpath('E:\myk\massive access\twc_massive_access_final_1115');

%% System Parameters
K = 500;     % the number of potential devices
Ka = 50;     % the number of active devices  
L = 4;       % the modulation order of PSK
M = 70;      % the length of spreading sequence
SNR_dB = 10; 
Sa = Ka/K; 
method=char('AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL','SWOMP','GSP','SAMP','Oracle LS','Gene-Aided OAMP');

% AMP algorithm paramemters
damp = 0;     % damping value
niter = 50;   % the number of iterations  
prior = 1;  % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;  % the threshold of activity detector
var_type = 1;  % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
T_set=[2:1:12];       % the number of consecutive OFDM symbols
testall = 8;                % the number of tested algorithms
ADEP = zeros(length(T_set), testall); 
BER = zeros(length(T_set), testall);
NMSE = zeros(length(T_set), testall);

%% Simulation
for alg_sel = [1:testall]
for sim = 1:Nsim
for mm = 1:length(T_set)   
    T=T_set(mm);
    
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
        case 1 % AMP-MMV
           [Xhat, lambda] = MMV_AMP_Gaussian_EM(Y, S_wave, damp, niter, 1e-8, 0);
           act_hat(mean(lambda,2)> ad_th) = 1;
       
        case 2 % OAMP-MMV-SSL
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP_MMV_SSL(Y, S_wave, damp, niter, prior, L);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;

        case 3 % OAMP-MMV-ASL
           Xhat = zeros(K,T);
           [Xhat_tmp, lambda] = OAMP_MMV_ASL(Y, S_wave, damp, niter, prior, L, var_type);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;
           
        case 4 %SWOMP
           epsilon = Ka/K/10^(SNR_dB/10);    
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
           samp_th=0.9;
           act_hat(sum(sum(abs(Xhat)>samp_ad,3),2)./T >= samp_th) = 1; 
           
       case 7 % Oracle LS
           S_oracle = S_wave(:,find(act_flag==1));
           Xhat_oracle = (S_oracle' * S_oracle)\(S_oracle'*Y);
           Xhat = zeros(K, T);
           Xhat(find(act_flag==1),:) = Xhat_oracle; 
           act_hat = act_flag;
 
       case 8 % Gene_aided OAMP
           Xhat = zeros(K,T);
           Xhat_tmp = Gene_aided_OAMP(Y, S_wave, damp, niter, prior, L, act_flag, SNR_dB, Sa);
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
    id_act_t = intersect(id_act,id_act_hat);
    id_act_f = setdiff(id_act,id_act_t);
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
        fprintf('%s, sim = %d, M = %d, SNR = %d dB, T=%d, ADEP = %4.8f, BER = %4.5f, NMSE = %7.2f\n', ...
                method(alg_sel,:), sim, M, SNR_dB, T, ADEP(mm,alg_sel)/sim, BER(mm,alg_sel)/sim, 10*log10(NMSE(mm,alg_sel)/sim));
    end
end
end
toc
end

Perform.ADEP = ADEP ./ Nsim;
Perform.BER = BER ./ Nsim;
Perform.NMSE = 10.*log10(NMSE./Nsim);

%% picture
MarkerSize = 8;
LineWidth = 1.5;
figure;
p1=semilogy(T_set,Perform.ADEP(:,1),'r-.+','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p2=semilogy(T_set,Perform.ADEP(:,2),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p3=semilogy(T_set,Perform.ADEP(:,3),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p4=semilogy(T_set,Perform.ADEP(:,4),'g-*','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p5=semilogy(T_set,Perform.ADEP(:,5),'k-s','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p6=semilogy(T_set,Perform.ADEP(:,6),'y-o','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('The number of OFDM symbols $T$ in one time slot','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
gcf_set = [p6,p4,p5,p1,p2,p3];
legend(gcf_set,{'SAMP','SWOMP','GSP','AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL'}...
      ,'location','southeast','interpreter','latex');

figure;
p1=semilogy(T_set,Perform.BER(:,1),'r-.+','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p2=semilogy(T_set,Perform.BER(:,2),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p3=semilogy(T_set,Perform.BER(:,3),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p4=semilogy(T_set,Perform.BER(:,4),'g-*','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p5=semilogy(T_set,Perform.BER(:,5),'k-s','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p6=semilogy(T_set,Perform.BER(:,6),'y-o','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on
p7=semilogy(T_set,Perform.BER(:,7),'b-','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on
p8=semilogy(T_set,Perform.BER(:,8),'-','LineWidth',LineWidth,'Color','[0.4 0.570 0.3410] ','MarkerSize',MarkerSize);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('The number of OFDM symbols $T$ in one time slot','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
gcf_set = [p6,p4,p5,p1,p7,p2,p3,p8];
legend(gcf_set,{'SAMP','SWOMP','GSP','AMP-MMV','Oracle LS','OAMP-MMV-SSL','OAMP-MMV-ASL','Gene-Aided OAMP'}...
       ,'location','southeast','interpreter','latex');








