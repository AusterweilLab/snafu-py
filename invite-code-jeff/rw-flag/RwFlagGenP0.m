function p0 = RwFlagGenP0(V)
p0 = rand(V,V);
p0(logical(diag(ones(1,V)))) = 0;
p0 = bsxfun(@rdivide, p0, sum(p0,2));
end
