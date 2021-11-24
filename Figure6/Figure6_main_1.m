%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 6)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.15
% OAMP-MMV-ASL for T = (5,10,15,20,25)
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
method = char('OAMP-MMV-ASL');

% AMP algorithm paramemters
damp = 0;      % damping value
Iiter = 50;    % the number of iterations  
prior = 1;     % 0/1 = Gaussian/MPSK prior
ad_th = 0.5;   % the threshold of activity detector
var_type = 2;  % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

%% Simulation Parameters
Nsim = 10000;                  % the number of simulation
M_set = [30:2:60];  % the length of spreading sequence
T_set = [5:5:25];
ADEP = zeros(length(M_set), length(T_set));     
BER = zeros(length(M_set), length(T_set));
NMSE = zeros(length(M_set), length(T_set));

%% Simulation
for i_T = 1:length(T_set)
    T = T_set(i_T);
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
            % OAMP-MMV-ASL
            Xhat = zeros(K,T);
            [Xhat_tmp, rou] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter, prior, L, var_type);
            Xhat(:,:) = Xhat_tmp(:,:,end);
            act_hat(mean(rou,2) > ad_th) = 1;        

            id_act_hat = find(act_hat==1);   % active device set (estimated)
            id_inact_hat = find(act_hat==0);
            Xhat(id_inact_hat,:) = 0;
            
            % ADEP
            ADEP(mm,i_T) = ADEP(mm,i_T) + sum(abs(act_hat-act_flag))/K;
            
            % NMSE  
            NMSE(mm,i_T) = NMSE(mm,i_T) + norm(Xhat-X,'fro')^2/norm(X,'fro')^2;

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
            BER(mm, i_T) = BER(mm, i_T) + Err_bit/(Ka*T*log2(L));
            BER_tmp = Err_bit/(Ka*T*log2(L)); 

            if mod(sim,100) == 0
                fprintf('%s, sim = %d, M = %d, T = %d, ADEP = %4.8f, BER = %4.5f, NMSE = %7.9f\n', ...
                        method, sim, M, T, ADEP(mm,i_T)/sim, BER(mm, i_T)/sim, 10*log10(NMSE(mm,i_T)/sim));
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
p1=semilogy(M_set,Perform.ADEP(:,1),'g-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p2=semilogy(M_set,Perform.ADEP(:,2),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p3=semilogy(M_set,Perform.ADEP(:,3),'m-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p4=semilogy(M_set,Perform.ADEP(:,4),'k-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p5=semilogy(M_set,Perform.ADEP(:,5),'r-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('Number of Measurements $M$','interpreter','latex','Fontsize',12);
ylabel('ADEP','interpreter','latex','Fontsize',12);
gcf_set = [p5,p4,p3,p2,p1];
legend(gcf_set,{'OAMP-MMV-ASL, $T$ = 5','OAMP-MMV-ASL, $T$ = 10','OAMP-MMV-ASL, $T$ = 15','OAMP-MMV-ASL, $T$ = 20','OAMP-MMV-ASL, $T$ = 25'},'location','northeast','interpreter','latex');

figure;
p1=semilogy(M_set,Perform.BER(:,1),'g-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p2=semilogy(M_set,Perform.BER(:,2),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p3=semilogy(M_set,Perform.BER(:,3),'m-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p4=semilogy(M_set,Perform.BER(:,4),'k-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
p5=semilogy(M_set,Perform.BER(:,5),'r-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);grid on; hold on
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('Number of Measurements $M$','interpreter','latex','Fontsize',12);
ylabel('BER','interpreter','latex','Fontsize',12);
legend('OAMP-MMV-ASL, $T$ = 5','OAMP-MMV-ASL, $T$ = 10','OAMP-MMV-ASL, $T$ = 15','OAMP-MMV-ASL, $T$ = 20','OAMP-MMV-ASL, $T$ = 25','location','southwest','interpreter','latex');




