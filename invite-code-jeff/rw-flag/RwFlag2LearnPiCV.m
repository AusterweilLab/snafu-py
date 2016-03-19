function [piHat, output] = RwFlag2LearnPiCV(V, D, cList, folds)
%RWFLAG2LEARNPI Summary of this function goes here
% IN
% OUT
% EX
% rng(0); folds = CVMakeFolds(20,5); 
% [piHat, output] = RwFlag2LearnPiCV(10, randi(10,20,3), [1,.1,.01], folds);
% piHat
% piHat =
%     0.2000    0.0667    0.0667    0.1000    0.0333    0.0333    0.1667    0.1333    0.1000    0.1000
% output
% output = 
%     cBest: 1
%     nllhMat: [5x3 double]
% output.nllhMat
% ans =
%    11.6461   15.7474   20.3057
%     8.2449    7.9532    7.9165
%     8.5326    8.3427    8.3203
%    10.0367   12.0338   14.3118
%     8.8735   10.2412   12.4146
%
% $Author: deltakam $	$Date: 2014/09/28 16:49:48 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

K = length(folds);
D1 = D(:,1); % just for improving the performance

% nllhMat(foldIdx, cIdx)
for foldIdx=1:K
  %- for each fold, estimate pi, then measure holdoutset likelihood
  trainIdx = CVGetTrainSet(folds, foldIdx);
  valiIdx = CVGetTestSet(folds, foldIdx);
  DTrain = D1(trainIdx,:);
  DVali = D1(valiIdx,:);

  for cIdx = 1:length(cList)
    cPi = cList(cIdx);
    pi = RwFlag2LearnPi(V, DTrain, cPi);
    
    nllh = RwFlag2NllhPi(DVali, pi);
    assert(~isnan(nllh));
    nllhMat(foldIdx, cIdx) = nllh;
  end
end

me = mean(nllhMat,1);
[minIdx, tieList] = EvalFindBest(me, 'min');
cPiBest = cList(minIdx);

%- decide cPiBest
piHat = RwFlag2LearnPi(V, D, cPiBest);

output.cBest = cPiBest;
output.nllhMat = nllhMat;
end
