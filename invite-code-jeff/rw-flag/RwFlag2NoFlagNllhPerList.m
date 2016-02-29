function [fList] = RwFlag2NoFlagNllhPerList(P, D)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: deltakam $	$Date: 2015/05/12 17:31:52 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2015

V = size(P,1);
% mask = RwFlag2UncMask(V);
% global NllhPerListParam
% NllhPerListParam.V = V;
% NllhPerListParam.D = [];
% NllhPerListParam.c = 0;
% NllhPerListParam.mask = mask;

% beta = RwFlag2UncP2BetaMat(P);

m = size(D,1);
fList = [];
for i=1:m
%   NllhPerListParam.D = D(i,:);
  fList(end+1) = RwFlag2NoFlagNLLH(P, D(i,:));
%   warning off MATLAB:singularMatrix;
%   fList(end+1) = RwFlag2UncObj(beta(mask), 'NllhPerListParam');
%   warning on MATLAB:singularMatrix;
end



end
