function out = EvalTestSignificanceMat(scoreMat, maxOrMin, alfa)
% scoreMat(i,j): i'th method, j'th trial
% alfa = 0.05 by default
%out.mean
%out.dev             : with prb (1-alfa), the mean is trapped in 'out.mean +(-) dev'.
%out.significanceMat : 'true' for those with statistical significance
%out.indistinctBest  : 'true' for those are indistinguishable from the best
%out.usedAlfa 
%--- Test Code
% cnt=0; 
% for i=1:1000, 
%   out = EvalTestSignificanceMat(randn(2,10)); cnt = cnt + all(out.indistinctBest); 
% end
% %- cnt should be around 950.


if (~exist('maxOrMin', 'var'))
  maxOrMin = 'max';
end
if (~exist('alfa','var'))
  alfa = 0.05;
end
assert(strcmp(maxOrMin, 'max') == true || strcmp(maxOrMin,'min') == true);

[nMethod, nTrial] = size(scoreMat);

meanAry = mean(scoreMat, 2)';
devAry = EvalCalcDeviation(scoreMat)';

significanceMat = false(nMethod, nMethod);

for i=1:nMethod
  for j=(i+1):nMethod
    [dev, thres, pval] = EvalTestSignificance(scoreMat(i,:), scoreMat(j,:));
    if (isnan(dev))
      tf = false;
    else
      tf = (dev > thres);
    end
    significanceMat(i,j) = tf;
    significanceMat(j,i) = tf;
  end
end

if (strcmp(maxOrMin, 'max'))
  [~,bestIdx] = max(meanAry); % the best  %TODO tie break with the least deviation
else
  [~,bestIdx] = min(meanAry); % the best  %TODO tie break with the least deviation
end

indistinctBest = (significanceMat(bestIdx,:) == 0);

out.mean = meanAry;
out.dev = devAry;
out.usedAlfa = alfa;
out.significanceMat = significanceMat;
out.indistinctBest = indistinctBest;
out.best = bestIdx;

end
