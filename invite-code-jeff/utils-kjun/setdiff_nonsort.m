function [S] = setdiff_nonsort(a,b)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
%  setdiff_nonsort([1 4 3 2], [1 2]);
%  ans =
%       4     3
% $Author: kjun $	$Date: 2014/11/30 13:52:04 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014


% a = [1 4 3 2] ; b = [1 2] ;
tf = ismember(a,b);
S = a(~tf);

end
