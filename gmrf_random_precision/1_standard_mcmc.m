% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% 'Standard MCMC' for 4.1 Random pattern precision matrices
%  


% Load model
clear()
load('example_1e6.mat');

% Initialize MCMC
accepts = 0;
mcmc_n = 1000;
step_size = .15;
samples = zeros(mcmc_n, 1);
X = -3;
X_log_p = -inf;

% Main algorithm MCMC here
for i = 1:mcmc_n

    % Display iteration number
    disp(strcat({'Running MCMC iteration '}, int2str(i)));
    
    % Propose new sample and evaluate log_likelihood
    X_prop = X + step_size*randn(1);
    k = exp(X_prop);
    Qk = 1./k*Q + k*speye(nn);
    R = chol(Qk);
    h_logdetQk = sum(log(diag(R)));
    X_prop_log_p = h_logdetQk - 0.5*yd'*Qk*yd;

    % Accept/Reject step
    if(log(rand()) < X_prop_log_p - X_log_p)
        X = X_prop;
        X_log_p = X_prop_log_p;
        accepts = accepts + 1;
    end;
    
    % Store
    samples(i) = X;
end;
