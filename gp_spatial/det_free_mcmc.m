% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% 'Det-free MCMC' for 4.2 Spatial Modelling of crime data
%     `


% Load model
clear();
load model;
load ra;

% Initialize MCMC
accepts = 0;
mcmc_n = 10000;
samples = zeros(mcmc_n, 3);
cov_e = [0.0326, 0.0060, 0.0013; 0.0060, 0.0247, -0.0024; 0.0013, -0.0024, 0.0013];
X = log([0.1310    0.0551    2.1123]);
X_log_p = -inf;
S = cov_mat(xd, exp(X));
[CG F] = pcg(S,yd, 1e-5, 100);
y_term = -0.5*yd'*CG + sum(-0.005.*X.*X);

% Main algorithm MCMC here
for i = 1:mcmc_n
    
    % Display iteration number
    disp(strcat({'Running MCMC iteration '}, int2str(i)));
    
    % Draw aux vars using multi-shift solver
    Z = multi_shift_sum(S, randn(nl, 1), shift, weight, 100, 1e-5);       
 
    % Propose new param values
    X_prop = mvnrnd(X, cov_e);
    S_prop = cov_mat(xd, exp(X_prop));
    
    % Evaluate log-likelihood terms (new and previous)
    [CG F] = pcg(S_prop,yd, 1e-5, 100);
    y_term_prop = -0.5*yd'*CG + + sum(-0.005.*X_prop.*X_prop);
    log_like_prop = -0.5*Z'*S_prop*Z + y_term_prop;
    log_like_prev = -0.5*Z'*S*Z + y_term;

    % Accept/Reject step
    if(log(rand()) < log_like_prop - log_like_prev)
        X = X_prop;
        S = S_prop;
        y_term = y_term_prop;
        accepts = accepts + 1;      
    end;
    
    % Store
    samples(i, :) = X;
    
end;
