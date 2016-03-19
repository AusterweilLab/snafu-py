function [ fold ] = makefolds(n, K)
% function [ fold ] = makefolds(n, K)
% make K folds out of n data (assuming index is from 1 to n)

  % make folds
  t = floor(n/K); % t: least # of item in a fold
  modu = mod(n, K);
  foldsizes = t * ones(K,1);
  foldsizes(1:modu) = foldsizes(1:modu) + 1;
  cumfs = cumsum(foldsizes);

  % fold{i} : set of index of data. fold{i} and fold{j} has no intersect if i ~= j
  fold = cell(K,1);
  randorder = randperm(n);

  fold{1} = randorder(1:cumfs(1));
  for k=2:K,  fold{k} = randorder((cumfs(k-1)+1):cumfs(k)); end

end

