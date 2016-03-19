function model = myLearnSvmLinear(X, y, C)
options = sprintf('-c %f -t 0 -q', C);
model = svmtrain(y,X, options);

end