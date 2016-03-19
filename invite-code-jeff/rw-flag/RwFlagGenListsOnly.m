function D = RwFlagGenListsOnly(pi,P,m,len)
% len: length of each list
V = length(pi);
if (~exist('len', 'var'))
  len = V;
end
assert (len <= V);

D = zeros(0,len);

for i=1:m
  
  %initial state
  s = find(mnrnd(1,pi) == 1);
  visited = false(1,V);
  visited(s) = true;
  
  componentSize = RwFlagCalcComponentSize(P, s);
  
  row = [s];
  while ( nnz(visited) ~= componentSize )
    s = find(mnrnd(1, P(s,:)));     % take a step
    if (visited(s) == false)
      row(end+1) = s;
    end
    visited(s) = true;              % update visited
  end
  
  D(end+1, 1:length(row)) = row;
end



end

