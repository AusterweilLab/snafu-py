function [nllhList] = RwFlag2NllhPerItem(pi, P, aList)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: deltakam $	$Date: 2014/10/19 16:24:32 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

row = ListsStrip(aList);
len = length(row);

nllhList = 0;
nllhList(1) = -log(pi(row(1)));

for k=1:len-1
  nonasb = row(1:k);          % nonabsorbing
  asb = row(k+1:end);         % absorbing

  Q = P(nonasb,nonasb);
  R = P(nonasb,asb);
  N = inv(eye(k) - Q);
  NR1 = N*R(:,1);
  prb = NR1(k);
  
  nllhList(k+1) = -log(prb);
end
end
