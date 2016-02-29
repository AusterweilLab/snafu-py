function subAry = randsamplesafe(ary, k)
if (length(ary) == 1)
  assert(k == 1);
  subAry = ary;
else
  subAry = randsample(ary, k);
end
end