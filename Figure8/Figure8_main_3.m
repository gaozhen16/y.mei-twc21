%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 12)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% SE of OAMP-MMV-SSL + SE of OAMP-MMV-ASL
% BER vs SNR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;
tic
addpath('E:\myk\massive access\twc_massive_access_final_1115\');

%% System Parameters
K = 500;             % the number of potential devices
Ka = 50;             % the number of active devices    
M = 70;              % the length of spreading sequence
L = 4;               % the modulation order of PSK
T = 10;              % the number of consecutive OFDM symbols
SNR_dB = 10;
Sa = Ka/K;
method=char('OAMP-MMV-SSL','OAMP-MMV-ASL');

% AMP algorithm paramemters
damp = 0;      % damping value
Iiter = 50;    % the number of iterations  
prior = 1;   % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;  % the threshold of activity detector

%% SE Parameters
testall = 2;                % the number of tested algorithms
Omega=pskmod(0:L-1, L, 0,'gray');
N = 1000;         
SNR_dB_set = [0:2:12];
MSE_se = zeros(Iiter,testall, length(SNR_dB_set)); 
BER_se = zeros(testall, length(SNR_dB_set)); 
Pe_se = zeros(testall, length(SNR_dB_set)); 
BER_SE_total = zeros(testall,length(SNR_dB_set));
for mm = 1:length(SNR_dB_set)
    SNR_dB = SNR_dB_set(mm);
    for alg_sel_se=1:testall        
        thegma2 = Ka/K/10^(SNR_dB/10);                                        

        % Generate Samples of X and noise Z
        x = zeros(K,T,N); 
        Xbit = zeros(K,T,N); 
        actu = zeros(Ka,N);        
        for n=1:N
            actu(:,n) = randperm(K,Ka);
            for j=1:T
                Xbit(actu(:,n),j,n) = Omega(ceil(rand(1,Ka)*L)).';
                x(actu(:,n),j,n) = Xbit(actu(:,n),j,n);
            end
        end       
        z=(randn(K,T,N)+1i*randn(K,T,N));
    
        % Calculate the posterior mean and MSE
        [miu, lambda, MSE_se(:,:,mm)] = OAMP_SE(x, z, Iiter, damp, L, M, N, thegma2, alg_sel_se, MSE_se(:,:,mm));
        act_hat_se_tmp=squeeze(mean(lambda,2));
    
        % BER
        for n=1:N
            act_hat_se = zeros(K,1); 
            act_hat_se(act_hat_se_tmp(:,n) > ad_th) = 1;
            id_act_hat = find(act_hat_se==1);
            id_act = actu(:,n);
            act_flag = zeros(K,1);
            act_flag(id_act) = 1;

            id_act_t = intersect(id_act,id_act_hat);
            id_act_f = setdiff(id_act,id_act_t);
            Pe_se(alg_sel_se, mm) = Pe_se(alg_sel_se, mm) + sum(abs(act_flag-act_hat_se))/K;
            if isempty(id_act_t)
                Err_bit = Ka*T*log2(L);
            else
                Xhat_demod = pskdemod(miu(id_act_t,:,n), L, 0,'gray');
                Xhat_demod = reshape(Xhat_demod,[],1);
                Xbit_BER = pskdemod(Xbit(id_act_t,:,n),L,0,'gray');
                Xbit_BER = reshape(Xbit_BER,[],1);
                if isempty(id_act_f)
                   Err_bit = sum(abs(de2bi(Xhat_demod,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all');
                else
                   Err_bit = sum(abs(de2bi(Xhat_demod,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all') + length(id_act_f)*T*log2(L);
                end
            end
            BER_se(alg_sel_se,mm) = BER_se(alg_sel_se,mm) + Err_bit/(Ka*T*log2(L));  
        end

        BER_SE_total(alg_sel_se, mm) = BER_se(alg_sel_se,mm)/N;
        fprintf('SE: %s, M = %d, SNR = %d dB, T=%d, Pe=%3.4f, BER=%4.7f, MSE = %7.9f\n', ...
              method(alg_sel_se,:), M, SNR_dB, T, Pe_se(alg_sel_se, mm)/N, BER_se(alg_sel_se, mm)/N, 10*log10(MSE_se(Iiter,alg_sel_se,mm)));
    end
end
toc

SE.BER_se = BER_SE_total;
SE.MSE_se = 10*log10(MSE_se);

%% picture
MarkerSize = 8;
LineWidth = 1.5;
%% plot with iteration
figure;
Iiter = 50;
semilogy(SNR_dB_set,SE.BER_se(1,:),'m-.<','LineWidth',LineWidth,'MarkerSize',MarkerSize); hold on;grid on;
semilogy(SNR_dB_set,SE.BER_se(2,:),'m-.>','LineWidth',LineWidth,'MarkerSize',MarkerSize); hold on;grid on;
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('SNR (dB)','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
legend('SE of OAMP-MMV-SSL', 'SE of OAMP-MMV-ASL','interpreter','latex');
