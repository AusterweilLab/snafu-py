function extraSample = spreadOutRemainder(mRemaining, n_l)
% extraSample: 1 by n_l array of integers (each are counts that sum to 'mRemaining')
% EX
% >> spreadOutRamainder(10, 3) 
% ans =
%      3     3     4
% >> spreadOutRamainder(10, 3) % 3 4 3 ( this is random!! )
% ans =
%      3     3     4

extraSample = double(idivide(int32(mRemaining), n_l, 'floor')) * ones(1,n_l);
remainder = mod(mRemaining, n_l);
idxList = randperm(n_l);
idxList = idxList(1:remainder);
extraSample(idxList) = extraSample(idxList) + 1;
assert(sum(extraSample) == mRemaining);
end
