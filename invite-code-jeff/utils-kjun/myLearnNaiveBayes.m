function model = myLearnNaiveBayes(X, y, alpha)

assert(length(alpha)== 1);
assert(alpha > 1);
nClasses = length(unique(y));
assert(min(y) == 1 && max(y) == nClasses);

y = ensureColVec(y);

[val,cnt] = countUnique(y);
assert(all(val==(1:nClasses)'));
py = ensureColVec(cnt./length(y));

for k=1:nClasses
  idx = y==k;
  numer = sum(X(idx,:),1) + alpha-1;
  
  pxy(:,k) = numer/sum(numer);
end

model.py = py;
model.pxy = pxy;
end
