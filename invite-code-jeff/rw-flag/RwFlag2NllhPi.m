function [nllh] = RwFlag2NllhPi(D, pi)
% DESCRIPTION 
% 
% IN 
% 
% OUT 
% 
% EX 
% 
% $Author: Jun $	$Date: 2014/09/28 19:54:27 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

assert(abs(sum(pi)-1) < 1e-6);

m=size(D,1);
nllh = 0;
for i=1:m
  nllh = nllh + (-log(pi(D(i,1))));
end
end
