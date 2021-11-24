%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% code: Compressive Sensing Based Joint Activity and Data Detection for
% Grant-Free Massive IoT Access (Figure 12)
% written by Yikun Mei (meiyikun@bit.edu.cn), Beijing Institute of Technology
% version: 2021.11.16
% OAMP-MMV-SSL + OAMP-MMV-ASL + SE of OAMP-MMV-SSL + SE of OAMP-MMV-ASL
% MSE vs iterations
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
var_type = 1; % the variance udpate used in OAMP-MMV-ASL algorithm, 1/2 = bayesian/non-bayesian

%% Simulation Parameters
Nsim = 1000;                  % the number of simulation
testall = 2;                % the number of tested algorithms
ADEP = zeros(testall);
BER = zeros(testall);
MSE = zeros(testall, Iiter);

%% Simulation
for alg_sel = 1:testall
for sim = 1:Nsim 
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
        case 1 % OAMP-MMV-SSL
           [Xhat, lambda] = OAMP_MMV_SSL(Y, S_wave, damp, Iiter, prior, L);   

        case 2 % OAMP-MMV-ASL
           [Xhat, lambda] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter, prior, L, var_type);   
    end

    % ADEP        
    act_hat(mean(lambda,2) > ad_th) = 1; 
    ADEP(alg_sel) = ADEP(alg_sel) + sum(abs(act_hat-act_flag))/K;

    % MSE
    for i_it=1:Iiter
       MSE(alg_sel,i_it) = MSE(alg_sel,i_it) + norm(Xhat(:,:,i_it)-X,'fro')^2/K/T;
    end

    id_act_hat = find(act_hat==1);
    id_inact_hat = find(act_hat==0);

    % BER
    Xhat(id_inact_hat,:) = 0;
    id_act_t = intersect(id_act,id_act_hat);
    id_act_f = setdiff(id_act,id_act_t);
    if isempty(id_act_t)
       Err_bit = Ka*T*log2(L);
    else
       Xhat_demod = pskdemod(Xhat(id_act_t,:,end), L, 0, 'gray');
       Xhat_demod = reshape(Xhat_demod,[],1);
       Xbit_BER = Xbit(id_act_t,:);
       Xbit_BER = reshape(Xbit_BER,[],1);
       Err_bit = sum(abs(de2bi(Xhat_demod,log2(L)) - de2bi(Xbit_BER ,log2(L))),'all') + length(id_act_f)*T*log2(L);
    end
    BER(alg_sel) = BER(alg_sel) + Err_bit/(Ka*T*log2(L));
    BER_tmp = Err_bit/(Ka*T*log2(L));

    if mod(sim,100) == 0
        fprintf('%s, sim = %d, M = %d, ADEP = %4.5f, BER = %4.5f, MSE = %7.9f\n', ...
                method(alg_sel,:), sim, M, ADEP(alg_sel)/sim, BER(alg_sel)/sim, 10*log10(MSE(alg_sel, end)/sim));
    end
end
end
Perform.ADEP = ADEP./Nsim;
Perform.BER = BER./Nsim;
Perform.MSE = 10*log10(MSE./Nsim);
disp('finish simulation of SE figure');

%% SE
Omega=pskmod(0:L-1, L, 0,'gray');
N = 1000;         
BER_se = zeros(testall, 1); 
ADEP_se = zeros(testall, 1); 
MSE_se = zeros(Iiter,testall);    
tic 
for alg_sel_se=1:testall   
    thegma2 = Ka/K/10^(SNR_dB/10);                                        
    
    % Generate Samples of X and noise Z
    x=zeros(K,T,N); 
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
    [miu, lambda, MSE_se] = OAMP_SE(x, z, Iiter, damp, L, M, N, thegma2, alg_sel_se, MSE_se);
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
        ADEP_se(alg_sel_se) = ADEP_se(alg_sel_se) + sum(abs(act_flag-act_hat_se))/K;
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
        BER_se(alg_sel_se) = BER_se(alg_sel_se) + Err_bit/(Ka*T*log2(L));  
    end
    
    fprintf('SE: %s, M = %d, SNR = %d dB, T=%d, ADEP=%3.4f, BER=%4.7f, MSE = %7.9f\n', ...
          method(alg_sel_se,:), M, SNR_dB, T, ADEP_se(alg_sel_se)/N, BER_se(alg_sel_se)/N, 10*log10(MSE_se(Iiter,alg_sel_se)));
end       
toc

Perform.ADEP_se =  ADEP_se/N;
Perform.BER_se = BER_se/N;
Perform.MSE_se = 10*log10(MSE_se);
disp('finish SE iteration');

%% picture
MarkerSize = 8;
LineWidth = 1.5;
%% plot with iteration
figure;
Iiter = 50;
xset = [1:10,11:2:Iiter];
plot(xset,reshape(Perform.MSE(1,xset),1,[]),'c--<','LineWidth',LineWidth,'MarkerSize',MarkerSize);hold on;grid on;
plot(xset,reshape(Perform.MSE_se(xset,1),1,[]),'k-.<','LineWidth',LineWidth,'MarkerSize',MarkerSize); hold on;grid on;
plot(xset,reshape(Perform.MSE(2,xset),1,[]),'c-->','LineWidth',LineWidth,'MarkerSize',MarkerSize);hold on;grid on;
plot(xset,reshape(Perform.MSE_se(xset,2),1,[]),'k-.>','LineWidth',LineWidth,'MarkerSize',MarkerSize); hold on;grid on;
set(gca,'position',[0.1 0.1 0.88 0.87]);
xlabel('Iteration','interpreter','latex','Fontsize',12);
ylabel('MSE','interpreter','latex','Fontsize',12);
legend('Simulated OAMP-MMV-SSL','Theoretical SE of OAMP-MMV-SSL','Simulated OAMP-MMV-ASL','Theoretical SE of OAMP-MMV-ASL','interpreter','latex');
