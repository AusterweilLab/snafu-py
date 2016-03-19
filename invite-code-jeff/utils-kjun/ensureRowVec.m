function v=ensureRowVec(v)
assert(min(size(v)) == 1);
v=reshape(v,1,length(v));
end