function predy = myPredNaiveBayes(X, model)
%model.py
%model.pxy

logpy = log(model.py);
logpxy = log(model.pxy);

predy = zeros(size(X,1),1);

for i=1:size(X,1)
  
  mat = bsxfun(@times, logpxy, X(i,:)');
  logp = logpy + sum(mat,1)';

  [~,maxIdx] = max(logp);
  
  predy(i) = maxIdx;
end

end
