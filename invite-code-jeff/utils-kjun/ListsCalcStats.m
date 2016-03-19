function out = ListsCalcStats(D, unknownIdx)
%- takes a special care for 'unknown item';
%- out.lenMean
%- out.lenMedian
%- out.nListWithDup
%- out.nDup
%- out.V
%- out.nVAppeared

if (unknownIdx == 0)
  existUnknown = false;
else
  existUnknown = true;
end

[m, maxLen] = size(D);
maxV = max(D(:));

nDupList = [];
lenList = [];
for ii=1:m
  row = D(ii,:);
  first0 = find(row == 0, 1, 'first');
  if (~isempty(first0))
    len = first0 - 1;
  else
    len = maxLen;
  end
  
  lenList(end+1) = len;  
  
  row = row(1:len);
  
  if (existUnknown)
    row(row == unknownIdx) = [];
  end
  
  nDup = length(row) - length(unique(row));
  
  nDupList(end+1) = nDup;
end

[members, counts] = countUnique(D(:));
if (members(1) == 0)
  members = members(2:end);
  counts = counts(2:end);
end

out.raw.VList = members;
out.raw.VCountList = counts;
out.raw.lenList = lenList;

out.lenMean = mean(lenList);
out.lenMedian = median(lenList);
out.lenMin = min(lenList);
out.lenMax = max(lenList);
out.m = m;
out.V = maxV;
out.nDup = sum(nDupList);
out.nListWithDup = nnz(nDupList);
out.VCountMean = mean(counts);
out.VCountMedian = median(counts);
out.VCountMin = min(counts);
out.VCountMax = max(counts);
out.nVAppeared = length(members);

end
