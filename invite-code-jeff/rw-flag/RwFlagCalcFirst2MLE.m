function PFirst2 = RwFlagCalcFirst2MLE(V,D,c)
% c is for smoothing.
% test:
% RwFlagCalcFirst2MLE(5, [1 2 0; 1 0 0],1)
% ans =
%          0    0.4000    0.2000    0.2000    0.2000
%     0.2500         0    0.2500    0.2500    0.2500
%     0.2500    0.2500         0    0.2500    0.2500
%     0.2500    0.2500    0.2500         0    0.2500
%     0.2500    0.2500    0.2500    0.2500         0


if (~exist('c','var'))
  c = 0;
end

C = c*ones(V,V);
C(logical(eye(V))) = 0;
for i=1:size(D,1)
  if (D(i,2) ~= 0)
    C(D(i,1), D(i,2)) = C(D(i,1), D(i,2)) + 1;
  end
end
PFirst2 = bsxfun(@rdivide, C, sum(C,2));
zeroRows = (sum(C,2) == 0);
PFirst2(zeroRows,:) = 1/(V-1);
PFirst2(logical(eye(V))) = 0;
end