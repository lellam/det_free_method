function Snm = cov_mat(xn, param)

ss = param(1);
ll = param(2);
tau = param(3);

xn = xn./ll;

Snm = pdist2(xn, xn, 'euclidean');
flags = Snm > 1.;
Snm(flags) = 0;
%Snm(~flags) = (1.- Snm(~flags)).^2;
Snm(~flags) = ss*(1.- Snm(~flags)).^4 .*((4.*Snm(~flags) + 1.));
%Snm(~flags) = 1./3*(1.- Snm(~flags)).^6 .* ((35.*Snm(~flags).^2) + (18.*Snm(~flags)) + 3);
Snm = sparse(Snm);
Snm = Snm + 1./tau*speye(size(xn, 1), size(xn, 1));



end
