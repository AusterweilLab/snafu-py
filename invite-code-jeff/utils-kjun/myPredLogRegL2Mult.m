function [pred, out] = myPredLogRegL2Mult(W, X)
X = [ones(size(X,1),1) X];
[~, nVars] = size(X);

out = X*[W zeros(nVars,1)];
[~, pred] = max(out, [], 2);
end
