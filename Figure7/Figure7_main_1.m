%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 7)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% OAMP-MMV-SSL + OAMP-MMV-ASL for lambda = (eq.(23), 0.01, 0.5, 0.99)
% ADEP and BER vs M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;
tic
addpath('E:\myk\massive access\twc_massive_access_final_1115\');

%% System Parameters
K = 500;     % the number of potential devices
Ka = 50;     % the number of active devices    
T = 10;      % the number of consecutive OFDM symbols
L = 4;       % the modulation order of PSK
SNR_dB = 10;  
Sa = Ka/K;
method = char('OAMP-MMV-SSL','OAMP-MMV-ASL');

% AMP algorithm paramemters
damp = 0;      % damping value
Iiter = 50;    % the number of iterations  
prior = 1;     % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;   % the threshold of activity detector
var_type = 2;  % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
M_set = [46:2:60];  % the length of spreading sequence
testall = 2;
lambda0_set = char('eq.23','0.01','0.5','0.99');
ADEP = zeros(length(M_set), length(lambda0_set), testall);     
BER = zeros(length(M_set), length(lambda0_set), testall);
NMSE = zeros(length(M_set), length(lambda0_set), testall);

%% Simulation
for i_test = 1:size(lambda0_set, 1)
    for alg_sel = 1:testall
        lambda0 = lambda0_set(i_test, :);
        for sim = 1:Nsim
            for mm = 1:length(M_set)
                M = M_set(mm);  % specific sample

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
                    case 1
                    % OAMP-MMV-SSL
                    Xhat = zeros(K,T);
                    if i_test == 1
                        [Xhat_tmp, lambda] = OAMP_MMV_SSL(Y, S_wave, damp, Iiter, prior, L);
                    else
                        [Xhat_tmp, lambda] = OAMP_MMV_SSL_test_para(Y, S_wave, damp, Iiter, prior, L, 'lambda', str2double(lambda0));
                    end
                    Xhat(:,:) = Xhat_tmp(:,:,end);
                    act_hat(mean(lambda,2) > ad_th) = 1;   
                    
                    case 2
                    % OAMP-MMV-ASL
                    Xhat = zeros(K,T);
                    if i_test == 1
                        [Xhat_tmp, lambda] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter, prior, L, var_type);
                    else
                        [Xhat_tmp, lambda] = OAMP_MMV_ASL_test_para(Y, S_wave, damp, Iiter, prior, L, var_type, 'lambda', str2double(lambda0));
                    end
                    Xhat(:,:) = Xhat_tmp(:,:,end);
                    act_hat(mean(lambda,2) > ad_th) = 1;   
                end

                id_act_hat = find(act_hat==1);   % active device set (estimated)
                id_inact_hat = find(act_hat==0);
                Xhat(id_inact_hat,:) = 0;

                % ADEP
                ADEP(mm,i_test,alg_sel) = ADEP(mm,i_test,alg_sel) + sum(abs(act_hat-act_flag))/K;

                % NMSE  
                NMSE(mm,i_test,alg_sel) = NMSE(mm,i_test,alg_sel) + norm(Xhat-X,'fro')^2/norm(X,'fro')^2;

                % BER
                id_act_t = intersect(id_act,id_act_hat);           % the set of correctly detected devices
                id_act_f = setdiff(id_act,id_act_t);               % the set of falsely detected devices
                Xhat_demod = pskdemod(Xhat(id_act_t,:), L, 0, 'gray');
                Xhat_demod = reshape(Xhat_demod,[],1);
                Xbit_BER = Xbit(id_act_t,:);
                Xbit_BER = reshape(Xbit_BER,[],1);                 % only calculating the error bits of active devices
                if isempty(id_act_t)
                   Err_bit = Ka*T*log2(L);
                else
                   Err_bit = sum(abs(de2bi(Xhat_demod,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all') + length(id_act_f)*T*log2(L);
                end
                BER(mm, i_test, alg_sel) = BER(mm, i_test, alg_sel) + Err_bit/(Ka*T*log2(L));
                BER_tmp = Err_bit/(Ka*T*log2(L)); 

                if mod(sim,100) == 0
                    fprintf('%s, sim = %d, M = %d, lambda = %s, ADEP = %4.8f, BER = %4.5f, NMSE = %7.9f\n', ...
                            method(alg_sel,:), sim, M, lambda0, ADEP(mm,i_test,alg_sel)/sim, BER(mm,i_test,alg_sel)/sim, 10*log10(NMSE(mm,i_test,alg_sel)/sim));
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
MarkerSize = 8;
LineWidth = 1.5;
figure;
semilogy(M_set,Perform.ADEP(:,1,1),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
semilogy(M_set,Perform.ADEP(:,1,2),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
semilogy(M_set,Perform.ADEP(:,2,1),'g-<','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.ADEP(:,2,2),'g->','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.ADEP(:,3,1),'r-<','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.ADEP(:,3,2),'r->','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.ADEP(:,4,1),'k-<','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.ADEP(:,4,2),'k->','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('Number of Measurements $M$','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
legend('OAMP-MMV-SSL, $\lambda^0 = eq.(23)$','OAMP-MMV-ASL, $\lambda^0 = eq.(23)$',...
       'OAMP-MMV-SSL, $\lambda^0 = 0.01$','OAMP-MMV-ASL, $\lambda^0 = 0.01$',...
       'OAMP-MMV-SSL, $\lambda^0 = 0.5$','OAMP-MMV-ASL, $\lambda^0 = 0.5$',...
       'OAMP-MMV-SSL, $\lambda^0 = 0.99$','OAMP-MMV-ASL, $\lambda^0 = 0.99$',...
       'location','northeast','interpreter','latex','fontsize',10.5);

figure;
semilogy(M_set,Perform.BER(:,1,1),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
semilogy(M_set,Perform.BER(:,1,2),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
semilogy(M_set,Perform.BER(:,2,1),'g-<','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.BER(:,2,2),'g->','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.BER(:,3,1),'r-<','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.BER(:,3,2),'r->','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.BER(:,4,1),'k-<','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
semilogy(M_set,Perform.BER(:,4,2),'k->','LineWidth',LineWidth,'MarkerSize',6);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('Number of Measurements $M$','interpreter','latex','Fontsize',12);
ylabel('$\mathrm{BER}$','interpreter','latex','Fontsize',12);
legend('OAMP-MMV-SSL, $\lambda^0 = eq.(23)$','OAMP-MMV-ASL, $\lambda^0 = eq.(23)$',...
       'OAMP-MMV-SSL, $\lambda^0 = 0.01$','OAMP-MMV-ASL, $\lambda^0 = 0.01$',...
       'OAMP-MMV-SSL, $\lambda^0 = 0.5$','OAMP-MMV-ASL, $\lambda^0 = 0.5$',...
       'OAMP-MMV-SSL, $\lambda^0 = 0.99$','OAMP-MMV-ASL, $\lambda^0 = 0.99$',...
       'location','northeast','interpreter','latex','fontsize',10.5);

