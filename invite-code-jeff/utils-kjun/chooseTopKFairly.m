function [idxList, threshold] = chooseTopKFairly(ary, k, sortDirection)
if (~exist('sortDirection', 'var'))
  sortDirection = 'descend';
end
assert(strcmp(sortDirection, 'descend') || strcmp(sortDirection, 'ascend'));


if (strcmp(sortDirection, 'descend'))
  bDescend = true;
else
  bDescend = false;
end


%--- exceptional case for k=1
if (k == 1)
  if (bDescend)
    threshold = max(ary);
  else
    threshold = min(ary);
  end
  tmp = find(ary==threshold);
  idxList = tmp(randi(length(tmp)));
  return;
end

%--- sort and find the threshold
sorted = sort(ary, sortDirection);
threshold = sorted(k);

% make sure things above threshold is sorted
if (bDescend)
  aboveThreshold = sorted > threshold;
else
  aboveThreshold = sorted < threshold;
end
aboveThresholdAry = sorted(aboveThreshold);

% old_sortedUnique = sort(unique(aboveThresholdAry, 'legacy'), sortOption); % this is optimized..
diffed = diff(aboveThresholdAry(:));
if (~isempty(diffed))
  sortedUnique = aboveThresholdAry([true;diffed~=0]);
else
  if (isempty(aboveThresholdAry))
    sortedUnique = [];
  else
    sortedUnique = aboveThresholdAry(1);
  end
end
% assert((isempty(old_sortedUnique) && isempty(sortedUnique)) || all(old_sortedUnique == sortedUnique));

idxList = [];
for i=1:length(sortedUnique)
  val = sortedUnique(i);
  ties = (ary == val);
  cnt = nnz(ties);
  indices = find(ties);
  indices = indices(randperm(cnt));
  
  idxList = [idxList indices];  
end
% [~, sidx] = sort(aboveThresholdAry, 'descend');
% aboveThresholdIdxList = find(aboveThreshold);
% aboveThresholdIdxList = aboveThresholdIdxList(sidx);

%--- choose those that sits right at the threshold
nChoose = k - length(idxList);
assert(nChoose >= 0);

tieIdxList = find(ary==threshold);

randIdx = randsample(length(tieIdxList), nChoose);

brokenTieIdxList = tieIdxList(randIdx);

%- append
idxList = [idxList, brokenTieIdxList];
assert(length(idxList) == k);


end
