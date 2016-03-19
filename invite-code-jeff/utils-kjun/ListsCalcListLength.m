function len = ListsCalcListLength(row)
%- compute zero-padded list length.
%- TEST
% ListsCalcListLength([1,2,3,0])
% ans =
%      3

maxLen = length(row);
first0 = find(row == 0, 1, 'first');
if (isempty(first0))
  len = maxLen;
else
  len = first0 - 1;
end

end