function [sortedMat, sortIdx] = mysortrows(mat, direction)
if (strcmp(direction, 'descend'))
  [sortedMat, sortIdx] = sortrows(-mat);
  sortedMat = -sortedMat;
elseif (strcmp(direction, 'ascend'))
  [sortedMat, sortIdx] = sortrows(mat);
else      assert(false);
end

end