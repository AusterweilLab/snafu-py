function [ fList ] = RwFlag2NllhPerListCon3(P, D)
%- this does not include the likelihood of the first item.
%- computes with RwFlagObj3
V = size(P,1);
mask = RwFlag2UncMask(V);
global NllhPerListParamCon
NllhPerListParamCon.V = V;
NllhPerListParamCon.D = [];
NllhPerListParamCon.c = 0;
NllhPerListParamCon.mask = logical(ones(V) - eye(V));

m = size(D,1);
fList = [];
for i=1:m
  NllhPerListParamCon.D = D(i,:);
%   warning off MATLAB:singularMatrix;
  fList(end+1) = RwFlagObj3(P(mask), 'NllhPerListParamCon');
%   warning on MATLAB:singularMatrix;
end

end