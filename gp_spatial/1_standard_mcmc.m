% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% 'Standard MCMC' for 4.2 Spatial Modelling of crime data
%     `


% Load model
clear();
load model;

% Initialize MCMC
accepts = 0;
mcmc_n = 10000;
samples = zeros(mcmc_n, 3);
cov_e = [0.0326, 0.0060, 0.0013; 0.0060, 0.0247, -0.0024; 0.0013, -0.0024, 0.0013];
X = log([0.1310    0.0551    2.1123]);
X_log_p = -inf;

% Main algorithm MCMC here
for i = 1:mcmc_n
    
    % Display iteration number
    disp(strcat({'Running MCMC iteration '}, int2str(i)));
    
    % Propose new sample
    X_prop = mvnrnd(X, cov_e);
    param = exp(X_prop);
    S = cov_mat(xd, param);
    L = chol(S);
    like = sum(log(diag(L))) + 0.5*yd'*(S \ yd);
    X_prop_log_p = like + sum(-0.005.*X_prop.*X_prop);
    
    % Accept/Reject step
    if(log(rand()) < X_prop_log_p - X_log_p)
        X = X_prop;
        X_log_p = X_prop_log_p;
        accepts = accepts + 1;  
    end;
    
    % Store
    samples(i, :) = X;
    
end;
