function [h_v_hat, iter_num] = SW_OMP_Algorithm(y_k_com, Phi, epsilon)
% This function is Simultaneous Weighted Orthogonal Matching Pursuit (SW-OMP) Algorithm.
% Inputs��
%   y_k_com��received signal y_k_com after combining K subcarriers
%   Phi��measurement matrix
%   Psi: dictionary matrix
%   epsilon���� is a tunable parameter defining the maximum error between the measurement and the received signal
%   D_w: an upper triangular matrix after accomplishing Cholesky factorization, which can be used to whiten process
% Outputs��
%   h_v_hat�� the sparse vector containing the complex channel gains
%   iter_num��number of iteration

[M_Ns, K] = size(y_k_com);    % size of y_k_com, M_Ns = M*Ns, K is the number of subcarrier
h_v_size = size(Phi,2);

% Compute the whitened equivalent observation matrix
Upsilon_w = Phi;

% Initialize the residual vectors to the input signal vectors and support estimate, where ..._com contain K columns
y_k_w_com = y_k_com;
Support_index_set = [];
r_k_com = y_k_w_com;
MSE = 2*epsilon;    % Pre-define MSE
iter_num = 0;       % Initialize the number of iteration
h_v_hat = zeros(h_v_size,K);

while (MSE > epsilon)
    % Distributed Correlation
    c_k_com = Upsilon_w'*r_k_com;
    
    % Find the maximum projection along the different spaces
    [~, index_p] = max(sum(abs(c_k_com),2));
    
    % Update the current guess of the common support
    Support_index_set = [Support_index_set; index_p];
    
    % Project the input signal onto the subspace given by the support using WLS
    xi_hat_com = Upsilon_w(:,Support_index_set)\y_k_w_com;
    
    % Update residual
    r_k_com = y_k_w_com - Upsilon_w(:,Support_index_set)*xi_hat_com;
    
    % Compute the current MSE
    MSE = 1/(M_Ns*K)*trace(r_k_com'*r_k_com);
    
    % Compte the number of iteration
    iter_num = iter_num + 1;
end

% assign estimated complex channel gains to the sparse vector
h_v_hat(Support_index_set,:) = xi_hat_com;

end