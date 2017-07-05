

load res_final.dat
xd = res_final(:, [1, 2]);
yd = res_final(:, 3);
load scale.dat
load weights.dat
load locs.dat
load o.mat
load xopt.mat
nl = length(yd);
load ra.m

accept_count = 0;
mcmc_n =10000;
burn = 2000;

samples = zeros(mcmc_n, 3);
accepts = zeros(mcmc_n);
times = zeros(mcmc_n);

t = cputime;

%X = log([0.1310    0.0551    2.1123]);
S = cov_mat(xd, exp(X));
[CG F] = pcg(S,yd, 1e-5, 100);
y_term = -0.5*yd'*CG + sum(-0.005.*X.*X);

for i = 1:mcmc_n+burn
    disp(i);
    % Sample aux vars
    
    w = randn(nl, 1);
%     L = chol(S);
%     Z = L \ w;
    Z = multi_shift_sum(S, w, shift, weight, 100, 1e-5);       
 
    % Update parameters
    log_like_prev = -0.5*Z'*S*Z + y_term;
    
    X_prop = mvnrnd(X, cov_e);
    S_prop = cov_mat(xd, exp(X_prop));
    [CG F] = pcg(S_prop,yd, 1e-5, 100, P, P');
    y_term_prop = -0.5*yd'*CG + + sum(-0.005.*X_prop.*X_prop);
    log_like_prop = -0.5*Z'*S_prop*Z + y_term_prop;
    
    log_U = log(rand());
    if(log_like_prop - log_like_prev > log_U)
        
        X = X_prop;
        S = S_prop;
        y_term = y_term_prop;
        
        if(i > burn)
            accept_count = accept_count + 1;
        end
    end;
    if(i > burn)
        samples(i-burn,:) = X;
        accepts(i-burn) = accept_count;
        times(i-burn) = cputime-t;
    end
    
end

accept_rate = accept_count/mcmc_n
