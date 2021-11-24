function [X_sic_bit,id_sic_act, nvar] = SICB_OAMP_MMV(Y,S_wave,damp,Iiter, prior, I_max, N_sic, L, hMod, hEnc, hDec, T_before, threshold, var_type, alg_type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% Y: Measurements
% S_wave: Meansurement matrix
% damp: Damping value
% Iiter: The maximum number of iterations
% prior: 0/1 for Gaussian/MPSK prior
% I_max: The maximum number of SIC iterations
% N_sic: The number of subtracted active devices in each SIC iteration
% L: Modulation order of MPSK
% hMod: Modulator
% hEnc: Channel Encoder
% hDec: Channel Decoder
% T_before; Length of data bits
% threshold: % The threshold of activity detector
% var_type: The udpate of variance, 1/2 = bayesian (line 4 in Algorithm 2)/non-bayesian ((27),[30])
% alg_type: 'SSL'/'ASL' for SICB-OAMP-MMV-SSL/SICB-OAMP-MMV-ASL
% Output:
% X_sic_bit£ºThe detected data btis
% id_sic_act: The estimated active device set
% nvar; The estimated noise variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialization
T_after = size(Y,2);             K = size(S_wave,2);
Xhat = zeros(K,T_after);
sic_flag = true;                 var_flag = true;
X_sic_bit = zeros(K,T_before);     id_sic_act = zeros(K,1);
sic_iter = 0;                           

% SIC
while sic_flag   
    if alg_type == 'SSL'
        [Xhat_tmp, lambda, nvartemp] = OAMP_MMV_SSL(Y, S_wave, damp, Iiter,prior ,L);   % SIC-OAMP-MMV-SSL
    else
        [Xhat_tmp, lambda, nvartemp] = OAMP_MMV_ASL(Y, S_wave, damp, Iiter,prior ,L, var_type);      % SIC-OAMP-MMV-ASL
    end
    Xhat(:,:) = Xhat_tmp(:,:,end);
    if var_flag
        nvar = nvartemp;
        var_flag = false;       % Pick the estimated noise variance in the first SIC iteration as the final estimation
    end
    
    % Pre-Activity Detection
    act_hat = zeros(K,1);
    act_hat(mean(lambda,2) > threshold) = 1;
    id_act_hat = find(act_hat == 1);
    
    % LLR for Subtraction
    Xbit_decode = zeros(length(id_act_hat), T_before);
    LLR_mean = zeros(length(id_act_hat), 1);
    for k = 1:length(id_act_hat)
       hDemod = comm.PSKDemodulator(L, 'BitOutput',true,...
       'DecisionMethod','Approximate log-likelihood ratio','PhaseOffset',0,'SymbolMapping','Gray','Variance',mean(mean(nvartemp,1),2));
       Shat = step(hDemod, Xhat(id_act_hat(k),:).');
       LLR_mean(k) = mean( sum( abs(Shat) ) );
       Xbit_decode(k,:) = step(hDec, -Shat);
    end
    [LLR_seq,seq] = sort(LLR_mean, 'descend');        % Sorting of the pre-detected active devices' LLR
    
    if length(LLR_seq) < N_sic
       for i_del=1:length(LLR_seq)     
           id_sic_act(id_act_hat(seq(i_del)),:) = 1;
           Xbit_del=Xbit_decode(seq(i_del),:);
           X_sic_bit(id_act_hat(seq(i_del)),:) = Xbit_del;
       end
       break;                                  % Stop Criteria 1:The number of pre-detected active devices is less than N_sic
    end
    
    id_del = zeros(N_sic,1);
    X_res = zeros(K,T_after);
    for i_del = 1:N_sic
       id_del(i_del) = seq(i_del);
       Xbit_del = Xbit_decode(id_del(i_del),:);
       encodedData = step(hEnc, Xbit_del.');
       X_res(id_act_hat(id_del(i_del)),:) = step(hMod, encodedData).';
    end                                         % Reconstruction of data matrix X
    acu_set = find(id_sic_act~=0);
    [~, posy] = find(id_act_hat(id_del).'== acu_set);                
    id_del(posy) = [];                                                % If some active devices are detected in the previous SIC iterations, then they will be removed
    id_sic_act(id_act_hat(id_del),:) = 1;
    X_sic_bit(id_act_hat(id_del),:) = Xbit_decode(id_del,:);        % Reserve the postion in support and data of subtracted devices
    Y = Y - S_wave * X_res;                              % Calculate residual
    
    sic_iter = sic_iter + 1;
    if sic_iter > I_max 
        break; 
    end                  % Stop Criteria 2£ºThe number of SIC iterations reaches I_max
end
end