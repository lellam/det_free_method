% Det-free method to simulate large Gaussian fields (2017)
%
% Louis Ellam 2017
%
% Effective Sample Size (ESS) Calculation

function e = ess(x)

N = length(x);
xh = mean(x);

g = zeros(size(x));
for k=1:N
  g(k) = 1/N * (x(1:(N-k))-xh)'*(x((1+k):end)-xh);
end

d = find((g/g(1))<0.01, 1);
e = 1/N * (g(1) + 2*sum(g(2:(d-1))));
e = g(1)/v;