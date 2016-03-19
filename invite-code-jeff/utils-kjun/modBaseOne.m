function [remainder] = modBaseOne(x, divider)
% (DESCRIPTION) 
%  modulo with base 1
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: kjun $	$Date: 2015/04/06 19:16:14 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2015

remainder = mod( x - 1, divider) + 1;

end
