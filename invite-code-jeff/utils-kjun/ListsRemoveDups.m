function DNew = ListsRemoveDups(D)

[m,maxLen] = size(D);
maxV = max(D(:));
DNew = zeros(0,maxLen);

maxLenNew = -inf;
for ii=1:m
  row = D(ii,:);  
  appeared = false(1,maxV);
  
  first0 = find(row == 0, 1, 'first');
  if (isempty(first0))
    len = maxLen;
  else
    len = first0 - 1;
  end
  
  row = row(1:len);
  
  rowNew = [];
  for j=1:len
    if (appeared(row(j)) == false)
      rowNew(end+1) = row(j);
      appeared(row(j)) = true;
    end
  end
  
  DNew(end+1, 1:length(rowNew)) = rowNew;
  
  maxLenNew = max(maxLenNew, length(rowNew));
end

%- compress DNew
tmp = DNew(:,maxLenNew+1:end);
assert(all(tmp(:) == 0));
DNew = DNew(:,1:maxLenNew);

end
