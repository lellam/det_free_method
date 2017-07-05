% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% 'Standard MCMC' for 4.3 GMRFs specified by a whitening matrix
%     `


% Load model
clear()
load('example_200.mat');

% Initialize MCMC
accepts = 0;
mcmc_n = 10000;
cov_e = [0.0002, 0.0000; 0.0000, 0.01];
samples = zeros(mcmc_n, 2);
X = [-2., -1.];
X_log_p = -Inf;

% Precomputation
Lap2 = Lap'*Lap;

% Main algorithm MCMC here
for i = 1:mcmc_n
    
    % Display iteration number
    disp(strcat({'Running MCMC iteration '}, int2str(i)));
    
    % Propose new sample and construct precision matrix
    X_prop = X + mvnrnd([0, 0]', cov_e);
    tau = exp(X_prop(1));
    kappa = exp(X_prop(2));
    
    % Compute new precision matrix from parameters
    Qk = 1./(kappa*kappa)*Lap2;
 
    % Compute log-det terms and keep cholesky of exp term
    L = chol(Qk, 'lower');
    log_det_Qk = 2.*sum(log(diag(L)));
    L = chol(Qk + tau*A'*A, 'lower');
    log_det_Qk_tau_AA = 2*sum(log(diag(L)));

    % Compute proposed log-density
    X_prop_log_p = tau*tau*(y_Y'*(A * (L' \ (L \ (A'*y_Y)))));
    X_prop_log_p = X_prop_log_p + log_det_Qk;
    X_prop_log_p = X_prop_log_p + y_n*log(tau);
    X_prop_log_p = X_prop_log_p - log_det_Qk_tau_AA;
    X_prop_log_p = X_prop_log_p - tau*y_Y'*y_Y;
    
    % Accept/Reject step
    if(log(rand()) < X_prop_log_p - X_log_p)
        X = X_prop;
        X_log_p = X_prop_log_p;
        accepts = accepts + 1;      
    end;
    
    % Store
    samples(i, :) = X;
    
end;
