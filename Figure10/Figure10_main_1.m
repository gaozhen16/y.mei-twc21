%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 10)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% AMP-MMV + OAMP-MMV-SSL + OAMP-MMV-ASL + SWOMP + GSP + SAMP + Oracle LS + 
% SICB-OAMP-MMV-SSL + SICB-OAMP-MMV-ASL
% ADEP and BER vs SNR, turbo coding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;
tic
addpath('E:\myk\massive access\twc_massive_access_final_1115\');

%% System Parameters
K = 500;             % the number of potential devices
Ka = 50;             % the number of active devices    
T_before = 30;       % the length of information bits
L = 4;               % the modulation order of PSK
M = 70;              % the length of spreading sequence 
Sa = Ka/K;
method=char('AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL','SWOMP','GSP', 'SAMP', 'SICB-OAMP-MMV-SSL', 'SICB-OAMP-MMV-ASL','Oracle LS','Gene-Aided OAMP');

% AMP algorithm paramemters
damp = 0;      % damping value
Iiter = 50;    % the number of iterations  
prior = 1;   % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;  % the threshold of activity detector
var_type = 1;  % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

% Turbo Encoder paramters
frmLen = T_before;
T_after = (T_before*3 + 12)/log2(L); % the length of coded bits
intrlvrIndices = randperm(frmLen);
hEnc = comm.TurboEncoder('InterleaverIndices',intrlvrIndices);
hMod = comm.PSKModulator(L, 'BitInput',true,'PhaseOffset',0,'SymbolMapping','Gray');
hDec = comm.TurboDecoder('InterleaverIndices',intrlvrIndices, ...
    'NumIterations',4);

% SIC parameters
I_max = 10;      % The maximum number of SIC iterations
N_sic = 10;       % The number of subtracted active devices in each SIC iteration

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
SNR_dB_set = [0:1:4,6:2:10]; 
testall = 10;                % the number of tested algorithms
ADEP = zeros(length(SNR_dB_set), testall);
BER = zeros(length(SNR_dB_set), testall);
NMSE = zeros(length(SNR_dB_set), testall);

%% Simulation
for alg_sel = [1:testall]
for sim = 1:Nsim
for i_SNR = 1:length(SNR_dB_set)   
    SNR_dB = SNR_dB_set(i_SNR);
    
    act_flag = zeros(K,1);
    act_flag(randperm(K,Ka)) = 1; % device activity indicator
    id_act = find(act_flag == 1); % active device set (true)
    
    X_data = zeros(T_after,K);
    Xbit = randi([0,1],K,T_before);
    for k=1:K
       encodedData  = step(hEnc, Xbit(k,:)');
       X_data(:,k) = step(hMod, encodedData);
    end
    X_data = X_data.';           % transmitted signals
    
    H = ones(K,1);       % considering the pre-equalization then channel effect is removed
    
    F = dftmtx(K) / sqrt(K);        
    P = eye(K);
    P = P(randperm(K,M),:);  
    S_wave = P*F;               % spreading code matrix

    H_tmp = H(:, ones(1,T_after));
    act_tmp = act_flag(:, ones(1,T_after));
    X = H_tmp.*act_tmp.*X_data;
    Y=awgn(S_wave*X,SNR_dB,'measured');     % received signals
    
    %% JADD
    act_hat = zeros(K,1);
    switch alg_sel
        case 1 % AMP-MMV
           [Xhat, lambda, nvar] = MMV_AMP_Gaussian_EM(Y, S_wave, damp, Iiter, 1e-8, 0);
           act_hat(mean(lambda,2) > ad_th) = 1;
              
        case 2 % OAMP-MMV-SSL 
           Xhat = zeros(K,T_after);
           [Xhat_tmp, lambda, nvar] = OAMP_MMV_SSL(Y, S_wave, damp, Iiter, prior ,L);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2) > ad_th) = 1;              
          
        case 3 % OAMP-MMV-ASL
           Xhat = zeros(K,T_after);
           [Xhat_tmp, lambda, nvar] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter, prior ,L, 2);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat(mean(lambda,2)> ad_th) = 1;             
           
        case 4 %SWOMP
           epsilon = Sa*10^(-SNR_dB/10);     
           [Xhat, iter_num] = SW_OMP_Algorithm(Y, S_wave, epsilon); 
           swomp_th = 0.01*max(max(abs(Xhat)));
           act_hat(mean(abs(Xhat),2) > swomp_th ) = 1;  
               
        case 5 % GSP
           [Xhat, supp, iter] = DMMV_SP(Y, S_wave, Ka);
           gsp_ad = 0.01*max(max(max(abs(Xhat))));
           gsp_th=0.9;
           act_hat(sum(sum(abs(Xhat)>gsp_ad,3),2)./T_after >= gsp_th) = 1; 
          
        case 6 % SAMP
           TH=0.005;
           [Xhat, supp, iter] = SAMP_distributed(Y, S_wave, TH);
           samp_ad = 0.01*max(max(max(abs(Xhat))));
           samp_th=0.9;
           act_hat(sum(sum(abs(Xhat)>samp_ad,3),2)./T_after >= samp_th) = 1; 
             
        case 7   %SICB-OAMP-MMV-SSL
           [X_sic_bit, id_sic_act, nvar] = SICB_OAMP_MMV(Y, S_wave, damp, Iiter,prior, I_max, N_sic ,...
                                                  L, hMod, hEnc, hDec, T_before, ad_th, var_type, 'SSL');
           act_hat = id_sic_act;
           % Recovery X for NMSE
           Xhat=zeros(T_after,K);
           for k=1:K
               encodedData  = step(hEnc, X_sic_bit(k,:)');
               Xhat(:,k) = step(hMod, encodedData);
           end
           Xhat= act_hat(:,ones(1,T_after)).*Xhat.';
    
        case 8   %SICB-OAMP-MMV-ASL
           [X_sic_bit, id_sic_act, nvar] = SICB_OAMP_MMV(Y, S_wave, damp, Iiter,prior, I_max, N_sic,...
                                                  L, hMod, hEnc, hDec, T_before, ad_th, var_type, 'ASL');
           act_hat = id_sic_act;
           % Recovery X for NMSE
           Xhat=zeros(T_after,K);
           for k=1:K
               encodedData  = step(hEnc, X_sic_bit(k,:)');
               Xhat(:,k) = step(hMod, encodedData);
           end
           Xhat= act_hat(:,ones(1,T_after)).*Xhat.';
           
       case 9 % Oracle LS
           S_oracle = S_wave(:,find(act_flag==1));
           Xhat_oracle = (S_oracle' * S_oracle)\(S_oracle'*Y);
           Xhat = zeros(K, T_after);
           Xhat(find(act_flag==1),:) = Xhat_oracle; 
           act_hat = act_flag;
           
      case 10 % Gene-Aided OAMP         
           Xhat = zeros(K,T_after);
           Xhat_tmp = Gene_aided_OAMP(Y, S_wave, damp, Iiter, prior, L, act_flag, SNR_dB, Sa);
           Xhat(:,:) = Xhat_tmp(:,:,end);
           act_hat = act_flag;
    end
    
    id_act_hat = find(act_hat==1);
    id_inact_hat = find(act_hat==0);
    Xhat(id_inact_hat,:) = 0;
    
    % ADEP
    ADEP(i_SNR,alg_sel) = ADEP(i_SNR,alg_sel) + sum(abs(act_hat-act_flag))/K;

    % NMSE  
    NMSE(i_SNR,alg_sel) = NMSE(i_SNR,alg_sel) + norm(Xhat-X,'fro')^2/norm(X,'fro')^2;
    
    % BER
    id_act_t = intersect(id_act,id_act_hat);
    id_act_f = setdiff(id_act,id_act_t);
    Xbit_decode = zeros(length(id_act_t),T_before);
    for k=1:length(id_act_t)
        if alg_sel<=3
            hDemod = comm.PSKDemodulator(L, 'BitOutput',true,...
            'DecisionMethod','Approximate log-likelihood ratio','PhaseOffset',0,'SymbolMapping','Gray','Variance',mean(mean(nvar,1),2));
    %                Shat = qamdemod(Xhat(id_act_t(k),:).', Nmod, 'gray','UnitAveragePower',true,'OutputType','approxllr','Noisevariance',mean(mean(nvar,1),2)); 
            Xhat_demod = step(hDemod,Xhat(id_act_t(k),:).');
            Xbit_decode(k,:) = step(hDec, -Xhat_demod);
        else if alg_sel<=6 || alg_sel>=9
            hDemod = comm.PSKDemodulator(L, 'BitOutput',true,...
            'DecisionMethod','Approximate log-likelihood ratio','PhaseOffset',0,'SymbolMapping','Gray','Variance',Sa*10^(-SNR_dB_set(i_SNR)/10));
    %                Shat = qamdemod(Xhat(id_act_t(k),:).', Nmod, 'gray','UnitAveragePower',true,'OutputType','approxllr','Noisevariance',Sa*10^(-SNR_dB(mm)/10));          
            Xhat_demod = step(hDemod,Xhat(id_act_t(k),:).');
            Xbit_decode(k,:) = step(hDec, -Xhat_demod);
            else
                Xbit_decode = X_sic_bit(id_act_t,:);   
                break;  
           end
        end 
    end
    Xbit_decode= reshape(Xbit_decode,[],1);
    Xbit_BER = Xbit(id_act_t,:);
    Xbit_BER = reshape(Xbit_BER,[],1);
    if isempty(id_act_t)
        Err_bit = Ka*T_before*log2(L);
    else
        Err_bit = sum(abs(de2bi(Xbit_decode,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all') + length(id_act_f)*T_before*log2(L);
    end
    BER(i_SNR, alg_sel) = BER(i_SNR, alg_sel) + Err_bit/(Ka*T_before*log2(L));
    BER_tmp = Err_bit/(Ka*T_before*log2(L));
    
    if mod(sim,100) == 0
        fprintf('%s, sim = %d, M = %d, SNR = %d dB,  ADEP = %4.5f, BER = %4.5f, NMSE = %7.9f\n', ...
                method(alg_sel,:), sim, M, SNR_dB, ADEP(i_SNR,alg_sel)/sim, BER(i_SNR, alg_sel)/sim, 10*log10(NMSE(i_SNR,alg_sel)/sim));
    end
end
end
toc
end

Perform.ADEP = ADEP ./ Nsim;
Perform.BER = BER ./ Nsim;
Perform.NMSE = 10.*log10(NMSE./Nsim);

%% picture
MarkerSize = 6;
LineWidth = 1.5;
figure;
p1=semilogy(SNR_dB_set,Perform.ADEP(:,1),'r-.+','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p2=semilogy(SNR_dB_set,Perform.ADEP(:,2),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p3=semilogy(SNR_dB_set,Perform.ADEP(:,3),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p4=semilogy(SNR_dB_set,Perform.ADEP(:,4),'g-*','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p5=semilogy(SNR_dB_set,Perform.ADEP(:,5),'k-s','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p6=semilogy(SNR_dB_set,Perform.ADEP(:,6),'y-o','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on
p7=semilogy(SNR_dB_set,Perform.ADEP(:,7),'b--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p8=semilogy(SNR_dB_set,Perform.ADEP(:,8),'b-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
gcf_set = [p6,p4,p5,p1,p2,p3,p7,p8];
legend(gcf_set,{'SAMP','SWOMP','GSP','AMP-MMV','OAMP-MMV-SSL','OAMP-MMV-ASL', 'SICB-OAMP-MMV-SSL', 'SICB-OAMP-MMV-ASL'}...
       ,'interpreter','latex','Fontsize',10.5);

figure;
p1=semilogy(SNR_dB_set,Perform.BER(:,1),'r-.+','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p2=semilogy(SNR_dB_set,Perform.BER(:,2),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p3=semilogy(SNR_dB_set,Perform.BER(:,3),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p4=semilogy(SNR_dB_set,Perform.BER(:,4),'g-*','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p5=semilogy(SNR_dB_set,Perform.BER(:,5),'k-s','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p6=semilogy(SNR_dB_set,Perform.BER(:,6),'y-o','LineWidth',LineWidth,'MarkerSize',6);grid on;hold on
p7=semilogy(SNR_dB_set,Perform.BER(:,7),'b--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p8=semilogy(SNR_dB_set,Perform.BER(:,8),'b-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p9=semilogy(SNR_dB_set,Perform.BER(:,9),'b-','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p10=semilogy(SNR_dB_set,Perform.BER(:,10),'-','LineWidth',LineWidth,'Color','[0.4 0.570 0.3410] ','MarkerSize',MarkerSize);grid on;hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
gcf_set = [p6,p4,p5,p1,p9,p2,p3,p7,p8, p10];
legend(gcf_set,{'SAMP','SWOMP','GSP','AMP-MMV','Oracle LS','OAMP-MMV-SSL','OAMP-MMV-ASL', 'SICB-OAMP-MMV-SSL', 'SICB-OAMP-MMV-ASL','Gene-Aided OAMP'}...
       ,'interpreter','latex');









