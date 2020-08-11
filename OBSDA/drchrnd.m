function r = drchrnd(a)
% take a sample from a dirichlet distribution
[v,k] = size(a);
r = gamrnd(a,1);
r = r ./ repmat(sum(r,1),v,1);