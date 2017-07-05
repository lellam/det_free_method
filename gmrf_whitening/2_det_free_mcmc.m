% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% 'Det-free MCMC' for 4.3 GMRFs specified by a whitening matrix
%     


% Load model
clear()
load('example_120.mat');

% Initialize MCMC
accepts = 0;
mcmc_n = 10000;
cov_e = [0.0002,  0.0000; 0.0000,    0.01];
samples = zeros([mcmc_n, 2]);
X = [-2., -1.];

% Precomputation
Lap2 = Lap'*Lap;

% Initialize vars
tau = exp(X(1));
kappa = exp(X(2));
Qk = 1./(kappa*kappa)*Lap2;
y_S_inv_y = y_Y'*(tau*y_Y - tau*tau*A*((Qk + tau*AA) \ (A' * y_Y)));

% Main algorithm MCMC here
for i = 1:mcmc_n
    
    % Display iteration number
    disp(strcat({'Running MCMC iteration '}, int2str(i)));
    
    % Sample auxiliary variables
    w = randn(x_n, 1);
    L = Lap \ (kappa*w);
    Y = A*L + 1./sqrt(tau) * randn(y_n, 1);
    Z = tau*Y - tau*tau*A*((Qk + tau*AA) \ (A' * Y));
    
    % Update previous likelihood
    z_S_z = Z'*(1./tau*Z + A*(Qk \ (A' * Z)));
    log_like_prev = -.5*(y_S_inv_y + z_S_z);
    
    % Propose new values
    X_prop = X + mvnrnd([0, 0]', cov_e);
    tau_prop = exp(X_prop(1));
    kappa_prop = exp(X_prop(2));
    Qk_prop = 1./(kappa_prop*kappa_prop)*Lap2;

    % New likelihood
    y_S_inv_y_prop = y_Y'*(tau_prop*y_Y - tau_prop*tau_prop*A*((Qk_prop + tau_prop*AA) \ (A' * y_Y)));
    z_S_z = Z'*(1./tau_prop*Z + A*(Qk_prop \ (A' * Z)));
    log_like_prop = -.5*(y_S_inv_y_prop + z_S_z);

    % Accept / Reject
    if(log(rand()) < log_like_prop - log_like_prev)
        X = X_prop;
        tau = tau_prop;
        kappa = kappa_prop;
        Qk = Qk_prop;
        y_S_inv_y = y_S_inv_y_prop;
        log_like_prev = log_like_prop;
        accepts = accepts + 1;
    end;
    
    % Store
    samples(i, :) = X;
    
end;
