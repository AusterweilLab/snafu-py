function mat = RwFlagUnc2Mask(V)
% generate a correct mask for beta parameter (no self transition, no identifiability issue)
mat = ones(V,V);
mat(logical(diag(ones(1,V)))) = 0;
mat = logical(mat);
end
