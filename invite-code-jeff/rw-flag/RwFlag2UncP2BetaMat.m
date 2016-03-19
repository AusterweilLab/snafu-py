function beta = RwFlagUnc2P2BetaMat(P)
%--- DESC
% turns P into beta matrix, making sure non-"neginf" members in each row has zero average.
% This way of normalizing keeps P the same while minimizing sum of squares of beta.
%--- TEST
% beta = RwFlagUnc2P2BetaMat(P)
% tmp = beta; tmp(isinf(tmp)) = 0; sum(tmp,2)/size(tmp,1) % must be zeros
assert (all((diag(P) < 1e-10)));
V = size(P,1);
nondiag = logical(1-diag(ones(1,V)));

beta = -inf(V,V);
beta(nondiag) = log(P(nondiag)); % this conatins -inf


rowMean = zeros(V,1);
for i=1:V
%   rowMean(i) = mean(beta(i,nondiag(i,:)));
  rowMean(i) = mean(beta(i, ~isinf(beta(i,:))));
end
beta = bsxfun(@minus, beta, rowMean);

end
