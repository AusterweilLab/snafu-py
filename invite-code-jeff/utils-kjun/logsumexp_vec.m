function s = logsumexp_vec(x)
% Returns log(sum(exp(x),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
% subtract the largest in each column
localy = max(x);
x = x - localy;
s = localy + log(sum(exp(x)));
locali = find(~isfinite(localy));
if ~isempty(locali)
  s(locali) = localy(locali);
end
end