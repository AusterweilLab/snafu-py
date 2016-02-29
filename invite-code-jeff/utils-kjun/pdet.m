function [ret] = pdet(A)
% (DESCRIPTION) 
%   computes the pseudo determinant
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: kjun $	$Date: 2015/07/07 00:50:24 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2015

s = svd(A);
idx = abs(s)>1e-7;
% nNonzeroSingularValues = nnz(idx)
ret = prod(s(idx));

end
