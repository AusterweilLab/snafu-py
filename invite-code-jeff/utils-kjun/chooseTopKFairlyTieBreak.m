function [idxList, threshold, thresholdForTieBreak] = chooseTopKFairlyTieBreak(ary, tieBreakAry, k, sortDirection)

if (strcmp(sortDirection, 'descend'))
  bDescend = true;
else
  bDescend = false;
end

assert(length(tieBreakAry) == length(ary));

%--- exceptional case for k=1
if (k == 1)
  if (bDescend)
    getBestFunc = @max;
  else
    getBestFunc = @min;
  end
  threshold = getBestFunc(ary);
  atThreshold = ary==threshold; % find threshold in ary
  thresholdForTieBreak = getBestFunc(tieBreakAry(atThreshold));% find threshold again in tieBreakAry
  tmp = find(atThreshold & tieBreakAry == thresholdForTieBreak);
  idxList = tmp(randi(length(tmp)));
  threshold = [threshold, thresholdForTieBreak];
  return;
end

%- set up matrix A: make our life easier using function sortrows
ary = ensureColVec(ary);
tieBreakAry = ensureColVec(tieBreakAry);
A = [ary tieBreakAry];

%--- sort and find the threshold
ASorted = mysortrows(A, sortDirection);
threshold = ASorted(k,:);

% make sure things above threshold is sorted
indices = ASorted(:,1) == threshold(1) & ASorted(:,2) == threshold(2);
aboveThreshold = 1:find(indices, 1, 'last');
ASortedTop = ASorted(aboveThreshold,:);

%- sortedUnique: sorted, unique rows.
sortedUnique = unique(ASortedTop, 'rows');
if (bDescend)
  sortedUnique = flipud(sortedUnique);
end

%- iteratively shuffle ties.
idxList = [];
for i=1:size(sortedUnique, 1)
  row = sortedUnique(i,:);
  ties = (A(:,1) == row(1) & A(:,2) == row(2));
  cnt = nnz(ties);
  indices = find(ties);
  indices = indices(randperm(cnt));
  
  idxList = [idxList indices'];  
end
% [~, sidx] = sort(aboveThresholdAry, 'descend');
% aboveThresholdIdxList = find(aboveThreshold);
% aboveThresholdIdxList = aboveThresholdIdxList(sidx);

assert(length(idxList) >= k);
idxList = idxList(1:k);

end
