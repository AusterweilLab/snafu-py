function s = logsumexp(x, dim)
% Returns log(sum(exp(x),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
if nargin == 1,
  % Determine which dimension sum will use
  dim = find(size(x)~=1,1);
  if isempty(dim), dim = 1; end
end
% subtract the largest in each column
localy = max(x,[],dim);
x = bsxfun(@minus,x,localy);
s = localy + log(sum(exp(x),dim));
locali = find(~isfinite(localy));
if ~isempty(locali)
  s(locali) = localy(locali);
end
end