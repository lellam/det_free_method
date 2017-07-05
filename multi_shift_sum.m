% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% Multishift solver that returns rational approximation
% N shifted CG-solves are approximately the cost of one
%
% Refer to:
% https://cdsweb.cern.ch/record/316892/files/9612014.pdf
%

function S = multi_shift_sum(A, b, shifts, weights, it_max, tol)


N = size(A, 1);
A = A + shifts(1)*speye(N);
shifts = (shifts - shifts(1))';
n_shift = length(shifts);
    
r = b;
p = b;
beta_old = 1.;
alpha = 0.;
c_cur = r' * r;
    
x_k = zeros(N, n_shift);    %Warning x_k stored horizontally!!!
p_k = repmat(b, 1, n_shift);
alpha_k=zeros(n_shift, 1);
beta_k=zeros(n_shift, 1);
xi_new_k = zeros(n_shift, 1);
xi_cur_k = ones(n_shift, 1);
xi_old_k = ones(n_shift, 1);
    
% CG Algorithm here
ii=0;
while ii < it_max
    
    a_p = A*p;
    beta_cur = -c_cur/(p'*A*p);
    denom = beta_cur*alpha*(xi_old_k-xi_cur_k) + beta_old*xi_old_k.*(1.-shifts*beta_cur);
    xi_new_k = beta_old.*xi_cur_k.*xi_old_k./denom;
    xi_cur_k = xi_cur_k + (xi_cur_k==0)*realmin;
    beta_k=beta_cur.*xi_new_k./xi_cur_k;
    x_k = x_k - repmat(beta_k', N, 1).*p_k;
    
    %Update residual, every 10 steps recompute from scratch
    if mod(ii,10) > 0
        r = r+beta_cur*a_p;
    else
        r = b-A*x_k(:, 1);
    end

    if norm(r) < tol
        if mod(ii,10) > 0                   %Recompute residual and norm
            r = b-A*x_k(:, 1);
            if norm(r) < tol                %Final termination check
                break
            end
        end
    end
                    
    c_new = r' * r;
    alpha = c_new/c_cur;
    p=r+alpha*p;
    alpha_k=alpha.*xi_new_k.*beta_k./(beta_cur.*xi_cur_k);
    p_k = repmat(xi_new_k', N, 1).*repmat(r, 1, n_shift) + repmat(alpha_k', N, 1).*p_k;
    
    % Update vars for next it
    xi_old_k=xi_cur_k;
    xi_cur_k=xi_new_k;
    beta_old=beta_cur;
    c_cur = c_new;
    ii = ii + 1;
end

% Non-convergence warning:
if(ii == it_max)
    disp('Warning max it reached');
end

% Return weighted sum
S = sum(x_k.*repmat(weights, N, 1), 2);

end