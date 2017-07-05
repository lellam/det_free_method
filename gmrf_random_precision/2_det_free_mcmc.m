% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% 'Det-free MCMC' for 4.1 Random pattern precision matrices
% 


% Load model
clear();
global yd;
global Q;
global nn;
load('example_1e4.mat');
load('ra');                         % Shifts and weights for sqrt-inverse

% Initialize MCMC
accepts = 0;
mcmc_n = 1000;
step_size = .15;
samples = zeros(mcmc_n, 1);
X = -3;

% Initialize vars
k = exp(X);
Qk = 1./k*Q + k*speye(nn);
y_term = -0.5*yd'*Qk*yd;

% Main algorithm MCMC here
for i = 1:mcmc_n
    
    % Display iteration number
    disp(strcat({'Running MCMC iteration '}, int2str(i)));
    
    % Draw aux vars using multi-shift solver
    Z = Qk * multi_shift_sum(Qk, randn(nn, 1), shift, weight, 100, 1e-5); 
    
    % Propose new param values
    X_prop = X + step_size*randn(1);
    k_prop = exp(X_prop);
    Qk_prop = 1./k_prop*Q + k_prop*speye(nn);
    
    % Evaluate log-likelihood terms (new and previous)
    [CG F] = pcg(Qk, Z, 1e-5, 100);
    X_log_p = y_term -0.5*Z'*CG;
    y_term_prop = -0.5*yd'*Qk_prop*yd;
    [CG F] = pcg(Qk_prop, Z, 1e-5, 100);
    X_prop_log_p = y_term_prop -0.5*Z'*CG;
    
    % Accept/Reject step
    if(log(rand()) < X_prop_log_p - X_log_p)
        X = X_prop;
        Qk = Qk_prop;
        y_term = y_term_prop;
        accepts = accepts + 1;
        
    end;
    
    % Store
    samples(i) = X;
end;
