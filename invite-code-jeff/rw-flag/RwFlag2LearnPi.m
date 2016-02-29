function [piHat] = RwFlag2LearnPi(V, D, cPi)
% Examples: 
% 
% 
% $Author: deltakam $	$Date: 2014/09/28 16:55:20 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

cnt = cPi * ones(1,V);
first = D(:,1);
for i=1:length(first)
  cnt(first(i)) = cnt(first(i)) + 1;
end
piHat = cnt / sum(cnt);

end
