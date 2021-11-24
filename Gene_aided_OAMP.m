function miu_storage = Gene_aided_OAMP(Y,S_wave,damp,iter,prior, L, act_flag, SNR_dB, Sa)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% Y: Measurements
% S_wave: Meansurement matrix
% damp: Damping value
% Iiter: The maximum number of iterations
% prior: 0/1 for Gaussian/MPSK prior
% L: Modulation order of MPSK
% act_flag: True activity information
% SNR_dB: Signal-to-noise ratio (dB)
% Sa: True sparsity ratio (Signal power)
% Output:
% miu_storage：The posterior mean of each iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prior Selection
if prior == 1 % 0/1 for Gaussian/MPSK
    Omega=pskmod([0:L-1], L, 0, 'gray');
    i_Omega=1:L;  
%     scatterplot(Omega);  
end
M = size(Y,1);
T = size(Y,2);
K = size(S_wave,2);

%EM initialization
c=-10:0.01:10;
c=exp(c);
Phinorm=1-0.5*erfc(-c/sqrt(2));
phinorm=1/sqrt(2*pi)*exp(-c.^2/2);
templambda0=(1+c.^2).*Phinorm-c.*phinorm;
labmda0=M/K*max((1-2*K*(templambda0)/M)./(1+c.^2-2.*(templambda0)));
thegma2=zeros(T,1);
THEGMA=zeros(T,1);
lambda=zeros(K,T);
for j=1:T
    thegma2(j) = Sa*10^(-SNR_dB/10);
%     thegma2(j)=norm(Y(:,j),2)^2/101/M;
%     THEGMA(j)=100/101*norm(Y(:,j),2)^2/norm(S_wave,'fro')^2;
%     lambda(:,j)=repmat(labmda0,K,1);
end      
lambda(find(act_flag==1),:) = 1;

% Other initializations
v = ones(T,1);                 u = zeros(K,T);
v_prev = zeros(T,1);           u_prev = zeros(K,T);
Z_0 = zeros(K,T);              Z_1 = zeros(K,T);          Z_2 = zeros(K,T);
miu_storage = zeros(K,T,iter);
for i=1:iter
%         for j=1:T
%             v(j) = norm(Y(:,j)-S_wave*u(:,j) ,2)^2/M-thegma2(j);
%         end
%         v = max(v, 1e-20);
        
        %moduleA , LMMSE        
        r = u+K/M*(S_wave'*(Y-S_wave*u));
        tao = (K-M)/M*v + K/M*thegma2;
        
        %moduleB , MMSE       
        if prior 
            for j=1:T
              Z_0(:,j)=sum(exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j))*Omega(i_Omega)))./tao(j)),2);
              Z_1(:,j)=sum(Omega(i_Omega).*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j))*Omega(i_Omega)))./tao(j)),2);
              Z_2(:,j)=sum(abs(Omega(i_Omega)).^2.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j))*Omega(i_Omega)))./tao(j)),2);            
            end
%             pai=1./(1+max((1-lambda)./Z_0, 1e-15));     
            miu=lambda.*Z_1./ Z_0;            
            gamma=lambda.*Z_2./ Z_0-abs((miu)).^2;  
            miu_storage(:,:,i) = miu; 
        else
            uX = (THEGMA./(tao+THEGMA)).'.*r;
            thegmaX = tao.*THEGMA./(tao+THEGMA);
            exponent = -(THEGMA./(THEGMA+tao)./tao).'.*abs(r).^2;
            pai = 1./(1+(1-lambda)./lambda.*((tao+THEGMA)./tao).'.*exp(exponent));
            miu = pai.*uX;
            miu_storage(:,:,i) = miu;
            gamma = pai.*thegmaX.'+pai.*(1-pai).*abs(uX).^2;   
        end

        gamma_mean = (sum(gamma,1)/K).';
        v = 1./(1./gamma_mean-1./tao);
        v = max(v, 1e-20);
        u = (v.').*(miu./(gamma_mean.')-r./(tao.'));


        %moduleA update
        u= (1-damp)*u+damp*u_prev;
        v = (1-damp)*v+damp*v_prev;
        v_prev = v;
        u_prev = u;
end
end