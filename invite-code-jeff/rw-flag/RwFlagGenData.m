function dset = RwFlagGenData(V,m,c,len)
% V: number of items
% m: the number of lists
% c: skewness parameter use 1 for default. 
% len: length of each list
if (~exist('len', 'var'))
  len = V;
end
assert (len <= V);

P = rand(V,V).^(c);
idx = logical(eye(V)); P(idx) = 0; % zero out diagonals
P = bsxfun(@rdivide, P, sum(P,2));
pi = ones(1,V)/V;

D = zeros(0,len);

for i=1:m
  
  %initial state
  s = find(mnrnd(1,pi) == 1);
  visited = false(1,V);
  visited(s) = true;
  
  row = [s];
  while ( nnz(visited) ~= len )
    s = find(mnrnd(1, P(s,:)));     % take a step
    if (visited(s) == false)
      row(end+1) = s;
    end
    visited(s) = true;              % update visited
  end
  
  D(end+1, :) = row;
end
dset.V = V;
dset.D = D;
dset.modelP = P;
dset.modelPi = pi;
end
