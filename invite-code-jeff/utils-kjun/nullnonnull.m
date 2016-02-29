function [null,nonnull] = nullnonnull(A)
% (DESCRIPTION) 
%   computes the null space and the nonnull space (=row space)
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: deltakam $	$Date: 2015/07/04 16:13:05 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2015

[m,n] = size(A);
[~,S,V] = svd(A,0);
if m > 1, s = diag(S);
  elseif m == 1, s = S(1);
  else s = 0;
end
tol = max(m,n) * max(s) * eps(class(A));
r = sum(s > tol);
null = V(:,r+1:n);
nonnull = V(:,1:r);

end
