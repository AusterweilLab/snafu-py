function printExpr(strExpr)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
%   aa = [1,2,3,4,5];
%   printexpr('length(aa)');
%   length(aa) = 
%     5
% $Author: deltakam $	$Date: 2014/10/19 13:32:07 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

fprintf('%s = \n', strExpr);
res = evalin('caller', strExpr);
disp(res);


end
