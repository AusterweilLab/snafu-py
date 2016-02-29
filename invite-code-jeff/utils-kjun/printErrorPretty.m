function printErrorPretty(errAry)
[nMethods, nRepeat] = size(errAry);

toPrint = [mean(errAry,2) std(errAry,0,2)/sqrt(nRepeat)];
fprintf('%12s %12s\n', 'sample mean','std error');
fprintf('%12f %12f\n', toPrint');
fprintf('%s\n', repmat('-',1,40));
me = mean(errAry, 2);
[~,argmax] = min(me);
fprintf('The best is method is %d\n', argmax);

for a=1:nMethods
  if (a == argmax)
    continue;
  end
  [dev, thres, pval]=getDeviation(0.05, errAry(a,:) - errAry(argmax,:));
  fprintf('method %2d vs method %2d: %d, pval=%f (dev=%12f, thres=%12f)\n', argmax, a, abs(dev)>thres, pval, dev, thres);
end
end