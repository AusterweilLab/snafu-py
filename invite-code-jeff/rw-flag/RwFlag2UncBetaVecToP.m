function P = RwFlag2UncBetaVecToP(beta, V)

betaMat = RwFlag2UncBetaVecToMat(beta, V);
P = bsxfun(@rdivide, exp(betaMat), sum(exp(betaMat),2));
end