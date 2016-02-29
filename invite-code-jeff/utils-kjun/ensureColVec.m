function v=ensureColVec(v)
assert(min(size(v)) == 1);
v=reshape(v,length(v),1);
end
