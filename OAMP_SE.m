function [miu, lambda, MSE_se] = OAMP_SE(x, z, Iiter, damp, L, M, N, thegma2, alg_sel_se, MSE_se)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% x: Samples
% z: Standard complex Gaussian noise
% Iiter: The maximum number of iterations
% damp: Damping value
% L: Modulation order of MPSK
% M: Number of measurements
% N: Number of Monte Carlo simulations
% thegma2; Noise variance
% alg_sel_se: 0/1 for OAMP-MMV-SSL/OAMP-MMV-ASL
% MSE_se: The predicted MSE (old)
% Output:
% miu：The posterior mean
% lambda: The posterior sparsity ratio
% MSE_se: The predicted MSE (new)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Omega=pskmod(0:L-1, L, 0,'gray');
i_Omega=1:L;  
K = size(x,1);        T = size(x,2);

% EM initialization
c=-10:0.01:10;
c=exp(c);
Phinorm=1-0.5*erfc(-c/sqrt(2));
phinorm=1/sqrt(2*pi)*exp(-c.^2/2);
temprou0=(1+c.^2).*Phinorm-c.*phinorm;
lambda0=M/K*max((1-2*K*(temprou0)/M)./(1+c.^2-2.*(temprou0)));
Sa=repmat(lambda0,K,N);
lambda = repmat(lambda0,K,T,N);

% Other initialization
Z_0 = zeros(K,T,N);     Z_1 = zeros(K,T,N);      Z_2 = zeros(K,T,N);
v = ones(Iiter+1,T);    v_prev = zeros(1,T);     u_prev = 0;
r=zeros(K,T,N);         tao = zeros(Iiter+1,T);   
mmse = zeros(Iiter,1);   

% Iteration
for i=1:Iiter   
    tao(i+1,:) = (K-M)/M*v(i,:) +K/M*thegma2;

    temp2 = zeros(K,T,N);
    for n=1:N
        for j=1:T                    
          r(:,j,n) = x(:,j,n) + sqrt(tao(i+1,j)/2)*z(:,j,n);              
          temp2(:,j,n) = sum(exp(-abs(r(:,j,n)-Omega(i_Omega)).^2/tao(i+1,j)),2)/L;
        end
    end

    if alg_sel_se == 2 % for OAMP-MMV-ASL
        temp1 = exp(-abs(r).^2./(tao(i+1,:)));  
        temp2_1 = max(temp2./temp1, 1e-15);
        temp4 = 1./(1+temp2_1);
        for n=1:N
            for k=1:K
                temp3 = ((1./Sa(k,n)-1)*prod(temp4(k,:,n)./(1-temp4(k,:,n)))./(temp4(k,:,n)./(1-temp4(k,:,n))));
                temp3 = max(temp3, 1e-15);
                lambda(k,:,n) = 1./(1+temp3);
            end
        end
     end

    for n=1:N
        for j=1:T  
            Z_0(:,j,n)=sum(lambda(:,j,n)/L.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j,n))*Omega(i_Omega)))./tao(i+1,j)),2);
            Z_1(:,j,n)=sum(lambda(:,j,n)/L.*Omega(i_Omega).*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j,n))*Omega(i_Omega)))./tao(i+1,j)),2);
            Z_2(:,j,n)=sum(lambda(:,j,n)/L.*abs(Omega(i_Omega)).^2.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j,n))*Omega(i_Omega)))./tao(i+1,j)),2);
        end
    end

    if alg_sel_se == 1 % for OAMP-MMV-SSL
        pai=1./(1+max((1-lambda)./Z_0,1e-15));               
    end           
    miu=Z_1./(1-lambda+Z_0);

    mmse(i) = norm(reshape(miu-x,[],1),'fro')^2/K/T/N;

    u = (tao(i+1,:)./(tao(i+1,:)-mmse(i))).*(miu -r.*mmse(i)./(tao(i+1,:)));
    for j=1:T
        v(i+1,j) = norm(reshape(u(:,j,:)-x(:,j,:),[],1),'fro')^2/K/N;
    end
    v(i+1,:) = (1-damp)*v(i+1,:) + damp*v_prev;
    v_prev = v(i+1,:);
    u = (1-damp)*u + damp*u_prev;
    u_prev = u;

    % update
    if alg_sel_se == 1             
        lambda = pai;        
        lambda_common = mean(lambda,2);
        lambda = repmat(lambda_common,1,T,1);
    else
        Sa = squeeze(mean(lambda,2));
    end   

    MSE_se(i,alg_sel_se) = MSE_se(i,alg_sel_se) + mmse(i);
end