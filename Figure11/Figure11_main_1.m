%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 11)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% SICB-OAMP-MMV-SSL + SICB-OAMP-MMV-ASL for N_sic = (5,10,20,25)
% ADEP and BER vs SNR, turbo coding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc; clear;
tic
addpath('E:\myk\massive access\twc_massive_access_final_1115\');

%% System Parameters
K = 500;             % the number of potential devices
Ka = 50;             % the number of active devices    
T_before = 30;       % the length of information bits
L = 4;               % the modulation order of PSK
M = 70;              % the length of spreading sequence 
Sa = Ka/K;
method=char('SICB-OAMP-MMV-SSL', 'SICB-OAMP-MMV-ASL');

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
% I_max = [4,5,10,20];      % The maximum number of SIC iterations
N_sic_set = [25,20,10,5];       % The number of subtracted active devices in each SIC iteration

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
SNR_dB_set = [0:1:4]; 
testall = 2;                % the number of tested algorithms
ADEP = zeros(length(SNR_dB_set), testall, length(N_sic_set));
BER = zeros(length(SNR_dB_set), testall, length(N_sic_set));
NMSE = zeros(length(SNR_dB_set), testall, length(N_sic_set));

%% Simulation
for i_Nsic = 1:length(N_sic_set)
    for alg_sel = [1:testall]
        for sim = 1:Nsim
            for i_SNR = 1:length(SNR_dB_set)   
                SNR_dB = SNR_dB_set(i_SNR);
                N_sic = N_sic_set(i_Nsic);
                I_max = 100/N_sic;
                
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
                    case 1   %SICB-OAMP-MMV-SSL
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

                    case 2   %SICB-OAMP-MMV-ASL
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
                end

                id_act_hat = find(act_hat==1);
                id_inact_hat = find(act_hat==0);
                Xhat(id_inact_hat,:) = 0;
                % ADEP
                ADEP(i_SNR,alg_sel,i_Nsic) = ADEP(i_SNR,alg_sel,i_Nsic) + sum(abs(act_hat-act_flag))/K;
                % NMSE  
                NMSE(i_SNR,alg_sel,i_Nsic) = NMSE(i_SNR,alg_sel,i_Nsic) + norm(Xhat-X,'fro')^2/norm(X,'fro')^2;

                % BER
                id_act_t = intersect(id_act,id_act_hat);
                id_act_f = setdiff(id_act,id_act_t);
                Xbit_decode = zeros(length(id_act_t),T_before);
                for k=1:length(id_act_t)
                    Xbit_decode = X_sic_bit(id_act_t,:);  
                end
                Xbit_decode= reshape(Xbit_decode,[],1);
                Xbit_BER = Xbit(id_act_t,:);
                Xbit_BER = reshape(Xbit_BER,[],1);
                if isempty(id_act_t)
                    Err_bit = Ka*T_before*log2(L);
                else
                    Err_bit = sum(abs(de2bi(Xbit_decode,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all') + length(id_act_f)*T_before*log2(L);
                end
                BER(i_SNR, alg_sel,i_Nsic) = BER(i_SNR, alg_sel,i_Nsic) + Err_bit/(Ka*T_before*log2(L));
                BER_tmp = Err_bit/(Ka*T_before*log2(L));

                if mod(sim,100) == 0
                    fprintf('%s, sim = %d, N_sic = %d, I_max = %d, SNR = %d dB,  ADEP = %4.5f, BER = %4.5f, NMSE = %7.9f\n', ...
                            method(alg_sel,:), sim, N_sic, I_max, SNR_dB, ADEP(i_SNR,alg_sel,i_Nsic)/sim, BER(i_SNR, alg_sel,i_Nsic)/sim, 10*log10(NMSE(i_SNR,alg_sel,i_Nsic)/sim));
                end
            end
        end
    toc
    end
end
Perform.ADEP = ADEP ./ Nsim;
Perform.BER = BER ./ Nsim;
Perform.NMSE = 10.*log10(NMSE./Nsim);

%% picture
MarkerSize = 6;
LineWidth = 1.5;
figure;
p1=semilogy(SNR_dB_set,Perform.ADEP(:,1,1),'g--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p2=semilogy(SNR_dB_set,Perform.ADEP(:,2,1),'g-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p3=semilogy(SNR_dB_set,Perform.ADEP(:,1,2),'b--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p4=semilogy(SNR_dB_set,Perform.ADEP(:,2,2),'b-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p5=semilogy(SNR_dB_set,Perform.ADEP(:,1,3),'r--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p6=semilogy(SNR_dB_set,Perform.ADEP(:,2,3),'r-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p7=semilogy(SNR_dB_set,Perform.ADEP(:,1,4),'k--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p8=semilogy(SNR_dB_set,Perform.ADEP(:,2,4),'k-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
gcf_set = [p1,p2,p3,p4,p5,p6,p7,p8];
legend(gcf_set,{'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (4, 25)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (4, 25)$',...
       'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (5, 20)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (5, 20)$',...
       'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (10, 10)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (10, 10)$',...
       'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (20, 5)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (20, 5)$'}...
       ,'interpreter','latex','fontsize',10.5);

figure;
p1=semilogy(SNR_dB_set,Perform.BER(:,1,1),'g--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p2=semilogy(SNR_dB_set,Perform.BER(:,2,1),'g-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p3=semilogy(SNR_dB_set,Perform.BER(:,1,2),'b--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p4=semilogy(SNR_dB_set,Perform.BER(:,2,2),'b-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p5=semilogy(SNR_dB_set,Perform.BER(:,1,3),'r--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p6=semilogy(SNR_dB_set,Perform.BER(:,2,3),'r-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p7=semilogy(SNR_dB_set,Perform.BER(:,1,4),'k--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
p8=semilogy(SNR_dB_set,Perform.BER(:,2,4),'k-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on;hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);axis([0 3 1e-6 1e-1]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
gcf_set = [p1,p2,p3,p4,p5,p6,p7,p8];
legend(gcf_set,{'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (4, 25)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (4, 25)$',...
       'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (5, 20)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (5, 20)$',...
       'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (10, 10)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (10, 10)$',...
       'SIC-OAMP-SSL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (20, 5)$', 'SIC-OAMP-ASL, $(I_{\mathrm{max}}, N^{\mathrm{sic}}) = (20, 5)$'}...
       ,'interpreter','latex','fontsize',10.5);








