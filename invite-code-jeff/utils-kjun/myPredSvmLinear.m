function predy = myPredSvmLinear(X, model)
predy = svmpredict(nan(size(X,1),1), X, model);
end

