function PWrong = RwFlagCalcWrongMLE(V,D,c)
% c is for smoothing.
% test:
% RwFlagCalcWrongMLE(5, [1 2 3 4; 1 0 0 0],1)
% ans =
%          0    0.4000    0.2000    0.2000    0.2000
%     0.2000         0    0.4000    0.2000    0.2000
%     0.2000    0.2000         0    0.4000    0.2000
%     0.2500    0.2500    0.2500         0    0.2500
%     0.2500    0.2500    0.2500    0.2500         0

if (~exist('c','var'))
  c = 0;
end

C = c*ones(V,V);
C(logical(eye(V))) = 0;
for i=1:size(D,1)
  endIdx = find(0 == D(i,:),1,'first');
  if (isempty(endIdx))
    len = size(D,2);
  else
    len = endIdx-1;
  end
  assert(len ~= 0);
  row = D(i,1:len);
  
  for j=1:len-1
    C(row(j), row(j+1)) = C(row(j), row(j+1)) + 1;
  end
end
PWrong = bsxfun(@rdivide, C, sum(C,2));
zeroRows = (sum(C,2) == 0);
PWrong(zeroRows,:) = 1/(V-1);
PWrong(logical(eye(V))) = 0;

end
