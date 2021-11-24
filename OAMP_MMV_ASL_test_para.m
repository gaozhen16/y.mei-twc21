function [miu_storage,lambda, thegma2] = OAMP_MMV_ASL_test_para(Y,S_wave,damp,Iiter,prior ,L, var_type, paratype, intvalue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% Y: Measurements
% S_wave: Meansurement matrix
% damp: Damping value
% Iiter: The maximum number of iterations
% prior: 0/1 for Gaussian/MPSK prior
% L: Modulation order of MPSK
% paratype: 'thegma2'/'lambda'
% intvalue: Initialization of selected parameter
% var_type: The udpate of variance, 1/2 = bayesian (line 4 in Algorithm 2)/non-bayesian ((27),[30])
% Output:
% miu_storage：The posterior mean of each iteration
% lambda: The posterior sparsity ratio
% thegma2; The estimated noise variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prior Selection
if prior == 1 % 0/1 for Gaussian/MPSK
    Omega=pskmod([0:L-1], L, 0,'gray');
    i_Omega=1:L;  
%     scatterplot(Omega);  
end
M = size(Y,1);      T = size(Y,2);       K = size(S_wave,2);

%EM Initialization
c=-10:0.01:10;
c=exp(c);
Phinorm=1-0.5*erfc(-c/sqrt(2));
phinorm=1/sqrt(2*pi)*exp(-c.^2/2);
templambda0=(1+c.^2).*Phinorm-c.*phinorm;
lambda0=M/K*max((1-2*K*(templambda0)/M)./(1+c.^2-2.*(templambda0)));
thegma2=zeros(T,1);
THEGMA=zeros(T,1);
if paratype(1) == 't'
    lambda = lambda0*ones(K,1);
    for j=1:T
    thegma2(j)=intvalue;
    THEGMA(j)=100/101*norm(Y(:,j),2)^2/norm(S_wave,'fro')^2;
    end  
else
    lambda = intvalue*ones(K,1);
    for j=1:T
    thegma2(j)=norm(Y(:,j),2)^2/101/M;
    THEGMA(j)=100/101*norm(Y(:,j),2)^2/norm(S_wave,'fro')^2;
    end   
end

%Other Initializations
xi = zeros(K,T);               
v = ones(T,1);                 u = zeros(K,T);            
u_prev = zeros(K,T);           v_prev = zeros(T,1);
miu_storage = zeros(K,T,Iiter);
Z_0 = zeros(K,T);              Z_1 = zeros(K,T);           Z_2 = zeros(K,T);
meanZ = zeros(M,T);            varz = zeros(M,T);
for i=1:Iiter
        if var_type == 2  % 2 is non-bayesian variance method
            for j=1:T
                v(j) = norm(Y(:,j)-S_wave*u(:,j) ,2)^2/M-thegma2(j);
            end
            v = max(v,1e-20);
        end
        
        % ModuleA , LMMSE        
        r = u+K/M*(S_wave'*(Y-S_wave*u));
        tao = (K-M)/M*v + K/M*thegma2;
        
        % ModuleB , MMSE 
        if prior 
          temp1 = exp(-abs(r).^2./tao.');  
          temp2 = zeros(K,T);
          for j=1:T
            temp2(:,j) = sum(exp(-abs(r(:,j)-Omega(i_Omega)).^2/tao(j)),2)/L;
          end             
          temp3 = 1./(1+max(temp2./temp1, 1e-15));
          for k=1:K                  
              temp4 = ((1./lambda(k)-1)*prod(temp3(k,:)./(1-temp3(k,:))).*((1-temp3(k,:))./temp3(k,:)));
              temp4 = max(temp4, 1e-15);
              xi(k,:) = 1./(1+temp4);
          end
            for j=1:T
              Z_0(:,j)=sum(xi(:,j)/L.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j))*Omega(i_Omega)))./tao(j)),2);
              Z_1(:,j)=sum(xi(:,j)/L*Omega(i_Omega).*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j))*Omega(i_Omega)))./tao(j)),2);
              Z_2(:,j)=sum(xi(:,j)/L*abs(Omega(i_Omega)).^2.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(r(:,j))*Omega(i_Omega)))./tao(j)),2);
            end
%             pai=1./(1+max((1-xi)./Z_0, 1e-15)); 
            miu=Z_1./((1-xi)+Z_0);
            miu_storage(:,:,i) = miu;
            gamma=Z_2./((1-xi)+Z_0)-abs((miu)).^2;  
        else 
             temp2 = zeros(K,T);
             for j=1:T
              temp2(:,j) = tao(j)/(THEGMA(j)+tao(j))*exp(abs(r(:,j)).^2*THEGMA(j)/tao(j)/(THEGMA(j)+tao(j)));
             end             
             temp3 = 1-1./(1+max(1./temp2, 1e-15));
            for k=1:K                   
                temp4 = ((1./lambda(k)-1)*prod(temp3(k,:)./(1-temp3(k,:))).*((1-temp3(k,:))./temp3(k,:)));
                temp4 = max(temp4, 1e-15);
                xi(k,:) = 1./(1+temp4);
            end
            uX = (THEGMA./(tao+THEGMA)).'.*r;
            thegmaX = tao.*THEGMA./(tao+THEGMA);
            exponent = -(THEGMA./(THEGMA+tao)./tao).'.*abs(r).^2;
            pai = 1./(1+(1-xi)./xi.*((tao+THEGMA)./tao).'.*exp(exponent));
            miu = pai.*uX;
            miu_storage(:,:,i) = miu;
            gamma = pai.*thegmaX.'+pai.*(1-pai).*abs(uX).^2;   
        end

        gamma_mean = (sum(gamma,1)/K).';
        v = 1./(1./gamma_mean-1./tao);
        v = max(v,1e-20);
        u = (v.').*(miu./(gamma_mean.')-r./(tao.'));


        %moduleA update
        u = (1-damp)*u+damp*u_prev;
        v = (1-damp)*v+damp*v_prev;
        v_prev= v;
        u_prev = u;

        %moduele C, EM update
        if prior == 0
            THEGMA=(sum(pai.*(thegmaX.'+abs(uX).^2))./sum(pai)).';    % Gaussian Variance of Signal
        end

        lambda = mean(xi,2);                         % Sparsity        

        sum_th=0;
        for j=1:T
            meanZ(:,j)=S_wave*miu(:,j);
            varz(:,j)=abs(S_wave).^2*gamma(:,j);
            
%             sum_th=sum(abs(Y(:,j)-meanZ(:,j)).^2+varz(:,j));
%             thegma2(j) = 1/M*sum_th;             %  Parallel Noise Variance
            
            sum_th=sum_th+sum(abs(Y(:,j)-meanZ(:,j)).^2+varz(:,j)); %  Single Noise Variance
        end
        thegma2(:) = 1/T/M*sum_th;
end