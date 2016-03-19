function out = myPredLogRegL2(w, X)
out = [ones(size(X,1),1) X]*w';
end
