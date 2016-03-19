function [bestIdx, tieList] = EvalFindBest(v, direction)

assert(strcmp(direction, 'min') || strcmp(direction, 'max'));
assert(size(v,1) == 1 || size(v,2) == 1);
v = v(:); % ensure column vector

if (strcmp(direction, 'min'))
  val = min(v);
else
  val = max(v);
end

tieList = find(v == val);
bestIdx = tieList(ceil(length(tieList)/2));

end