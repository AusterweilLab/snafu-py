function [state, seeds] = GenSeeds2(metaSeedOrState, m, n)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: kjun $	$Date: 2014/11/02 23:33:56 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014
if (exist('m', 'var') && ~exist('n', 'var'))
  n = m;
  m = 1;
elseif (~exist('m','var'))
  m = 1;
  n = 1;
end
oldState = rng(metaSeedOrState);
seeds = randi(2^31-1,m,n, 'int32');
state = rng(oldState);
end
