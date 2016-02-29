function [outAry] = spreadOutRemainder_append(mRemaining, ary)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% spreadOutRemainder_append(10, [3,4,3,3]) % output has randomness
% ans =
%      6     5     6     6
% spreadOutRemainder_append(10, [3,4,3,3])
% ans =
%      5     6     6     6
% 
% $Author: kwang-sungjun $	$Date: 2015/10/01 11:34:07 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2015

minVal = min(ary);
maxVal = max(ary);
assert(maxVal - minVal <= 1);

if (minVal ~= maxVal)
  minValIdx = ary == minVal;
  nMinVal = nnz(minValIdx);
  
  if (mRemaining > nMinVal);
    ary(minValIdx) = ary(minValIdx) + 1;
    mRemaining = mRemaining - nMinVal;
  else
    %- very complicated here, but what I am doing is to
    %- increase those guys with `minVal' by 1, but chose those
    %- uniformly randomly
    aa = find(minValIdx);
    myPerm = randperm(length(aa));
    tmpIdx = aa(myPerm(1:mRemaining));
    ary(tmpIdx) = ary(tmpIdx) + 1;
    outAry = ary;
    return;
  end
end

appendAry = spreadOutRemainder(mRemaining, length(ary));

outAry = ary + appendAry;

assert(sum(outAry) == mRemaining + sum(ary));
end
