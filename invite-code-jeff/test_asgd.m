%- This code finds the best ssGamma0 (gamma_0 in the paper) for each
%- c value since the best gamma_0 value changes with different c value.

dbg = true;
cIdx = 3;

cList = 10.^[1, .5, 0, -.5, -1, -1.5, -2];
ssGamma0List = 10.^[2, 1.5, 1, 0.5, 0, -0.5, -1]; % step size gamma0
mTrain = 100;
nEpoch = 20; %10

% seedFolds = 1;
seedP0 = 1
seedPerm = 2
seedAlgo = 3
seedTestSplit = 10

%- toy data
%V = 100;
%len = 20;
%dset = RwFlagGenData(V,1000,1, len);
V = 15;
len = 15;
dset = RwFlagGenData(V,1000,1, len);
D = dset.D;
m = size(D,1);
mask = RwFlag2UncMask(V);

%- split train/test, filter D so that it doesn't have test portion
mTest = round(m*.1);
mNonTest = m - mTest;
oldState = rng(seedTestSplit); tmp = randperm(m); rng(oldState);
testIdx = tmp(1:mTest);
trainIdx = setdiff(1:m, testIdx);
fprintf('mNonTest = %d, mTest = %d\n', mNonTest, mTest);
printExpr('testIdx(1:10)');
fprintf('Selecting train data ...\n');
D = D(trainIdx,:);
m = size(D,1)

%- compute list stats, remove dups, then compute the stats again.
DStats = ListsCalcStats(D, false)
D = ListsRemoveDups(D);
DStats = ListsCalcStats(D, false)
mAll = size(D,1);

%----- In case of debugging
cOrig = cList(cIdx);
if (dbg)
  ssGamma0List = 10.^[-0.5, -1];
  mTrain = 50;
  nEpoch = 10;
  c = (cOrig / mAll) * mTrain
else
  printExpr('cList(cIdx)');
  fprintf('mAll = %d, mTrain = %d\n', mAll, mTrain);
  c = (cOrig / mAll) * mTrain
end

% set step size parameter
ssA = cOrig / mAll;

%----- Testing purpose.
mTrain
oldState = rng(seedPerm);
permIdx = randperm(size(D,1)); % randomization
rng(oldState);
DTrain = D(permIdx(1:mTrain),:);
DVali = D(permIdx(mTrain+1:2*mTrain),:);

%----- p0
oldState = rng(seedP0);
p0 = RwFlag2GenP0(V);
rng(oldState);
beta0 = RwFlag2UncP2BetaMat(p0);

%----- Run LBFGS
global ObjParam
ObjParam.V = V;
ObjParam.D = DTrain;
ObjParam.c = cOrig;
ObjParam.mask = mask;

lbfgsOpt = struct('maxIts', 4000, 'x0', beta0(mask), 'printEvery', 20);
lb = -inf(size(lbfgsOpt.x0)); ub = inf(size(lbfgsOpt.x0));
tic;
[beta, trainObjPSACBatch, outputBatch] = lbfgsb(@RwFlag2UncObj, lb, ub, lbfgsOpt);
fprintf('lbfgsb function count = %d\n', outputBatch.totalIterations);
toc
beta = RwFlag2UncBetaVecToMat(beta, V);
PSACBatch = bsxfun(@rdivide, exp(beta), sum(exp(beta),2));

valiNllhPSACBatch = sum(RwFlag2NllhPerList(PSACBatch, DVali))

%----- Run ASGD
% ssGamma0List
% ssAList
trainObjMat = [];
valiNllhMat = [];
t = nan;
ssGamma0List
for gIdx=1:length(ssGamma0List)
  %--- prepare option struct
  opt = FMinUncAsgdOptions();
  opt.nEpoch = nEpoch;
  opt.objC = c;
  opt.dbg = true;
  if (opt.dbg == true)
    opt.dbgData = DVali;
    opt.dbgInterval = mTrain;
  end
  opt.stepsizeGamma0 = ssGamma0List(gIdx);
  opt.stepsizeA = ssA;
  opt.stepsizeC = 3/4;
  opt.mask = mask;
  opt

  %----- Run optimization
  %- could use the count estimate to start the optimization
  %- but for debugging purpose, I will not use it now.
  rng(seedAlgo);
  tic;
  [beta,output] = FMinUncAsgd(@RwFlag2UncObj, beta0(mask), V, DTrain, opt);
  toc
  output
  PSAC = RwFlag2UncBetaVecToP(beta, V);

  if (output.exitflag == 0)
    fprintf('Hold-out set likelihoods = \n');
    disp(output.dbgNllhVali(end));
    nllhPSAC = output.dbgNllhVali(end);

    if (isnan(t))
      t = output.dbgObjTimeList;
    end
    trainObjMat(1:length(t), gIdx) = output.dbgObjTrain;
    valiNllhMat(1:length(t), gIdx) = output.dbgNllhVali;
  else
    fprintf('ASGD Optimization Failed.\n');
    nllhPSAC = inf;
    trainObjMat(1:length(t), gIdx) = inf;
    valiNllhMat(1:length(t), gIdx) = inf;
  end
end

epoch = t/mTrain;
%- trainObjPSACBatch, trainObjMat(:, gIdx)
%- valiNllhPSACBatch, valiNllhMat(:, gIdx)

fNameMat = 'result-asgd';
fprintf('Saving to %s\n', fNameMat);
save(fNameMat);


