function betaMat = RwFlag2UncBetaVecToMat(beta, V)
betaMat = zeros(V,V);
betaMat(logical(eye(V))) = -inf;
mask = RwFlag2UncMask(V);
betaMat(mask) = beta;
end