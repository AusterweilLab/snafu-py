function [ fList ] = RwFlag2NllhPerList(P, D)
%- this does not include the likelihood of the first item.
V = size(P,1);
mask = RwFlag2UncMask(V);
global NllhPerListParam
NllhPerListParam.V = V;
NllhPerListParam.D = [];
NllhPerListParam.c = 0;
NllhPerListParam.mask = mask;

beta = RwFlag2UncP2BetaMat(P);

m = size(D,1);
fList = [];
for i=1:m
  NllhPerListParam.D = D(i,:);
  warning off MATLAB:singularMatrix;
  fList(end+1) = RwFlag2UncObj(beta(mask), 'NllhPerListParam');
  warning on MATLAB:singularMatrix;
end

end