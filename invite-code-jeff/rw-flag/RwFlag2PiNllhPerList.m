function [nllhList] = RwFlag2PiNllhPerList(D, pi)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% rng(0);RwFlag2PiNllhPerList(randi(4,10,3), [.4 .3 .2 .1])'
% ans =
%     2.3026    2.3026    0.9163    2.3026    1.6094    0.9163    1.2040    1.6094    2.3026    2.3026
%
% $Author: kjun $	$Date: 2014/10/17 19:23:24 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

assert(abs(sum(pi)-1) < 1e-6);

nllhList = -log(pi(D(:,1)));
nllhList = nllhList(:);

end
