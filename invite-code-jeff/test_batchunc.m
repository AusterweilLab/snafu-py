clear;
format compact;

mList = [10, 40];
nTry = 1;
tryIdxList = 1:nTry;

bDebug = false

seedFolds = 1
seedAlgo = 2
seedP = 3

%--- Results I want
V = 25
oldState = rng(seedP);
modelP = RwFlag2GenPToyGrid(V);
componentSize = RwFlag2CalcComponentSize(modelP, 1);
assert(componentSize == V);
rng(oldState);

%- prepare variables
% cList = 10.^[1, .5, 0, -.5, -1, -1.5, -2];
cList = 10.^[1,0,-1]; warning('DEBUG');
K = 5;
mask = RwFlag2UncMask(V);
dummyPi = ones(1,V)/V;

% mList = [10, 20, 40];
% mList = 160;
% mList = 320;
mList
tryIdxList = 1:nTry


% %--- Starting loop
% if (bDebug)
%   warning('DEBUG');
% %   cList = 10.^[2,0,-2];
%   cList = 10.^[2, 0, -2];
%   mList = [10, 20];
%   nTry = 2;
%   tryIdxList = 1:nTry;
% end

foldSeedList = GenSeeds(length(mList), seedFolds);
trySeedList = GenSeeds(nTry, seedAlgo);

cMat = []; % (tryIdx, mIdx, estIdx)
PCell = cell(0,0,0); % (tryIdx, mIdx, estIdx)
trainNllhMat = []; % (tryIdx, mIdx, estIdx)

estNameList = {'PInvite', 'PRW', 'First2'}
nEst = length(estNameList);
foldSeedList
trySeedList
cList
K
fprintf('dummyPi is just so I can compute loglik; doesn''t matter as long as dummyPi > 0\n');
dummyPi

for tryIdx = tryIdxList
  fprintf('#----- tryIdx = %d -------------------------------------------------\n', tryIdx);
  trySeed = trySeedList(tryIdx)
  rng(trySeed); 
  
  DGenerated = RwFlag2GenListsOnly(dummyPi, modelP, max(mList));
  
  for mIdx=1:length(mList)
    fprintf('#----- mIdx = %d -----------------\n', mIdx);
    m = mList(mIdx)
    seed = foldSeedList(mIdx)

    DAll = DGenerated(1:m,:);

    oldState = rng(seed);
    folds = CVMakeFolds(m,K);
    rng(oldState);

    global ObjParam
    ObjParam.V = V;
    ObjParam.mask = mask;

    nllhMat = zeros(K, length(cList), nEst);
    for cIdx = 1:length(cList)
      fprintf('#--- cIdx = %d\n', cIdx);
      c = cList(cIdx)

      for foldIdx=1:K
        fprintf('#- fold %d\n', foldIdx);
        trainIdx = CVGetTrainSet(folds, foldIdx);
        valiIdx = CVGetTestSet(folds, foldIdx);
        DTrain = DAll(trainIdx,:);
        DVali = DAll(valiIdx,:);

        %--- Learn PWrong, measure nllh
        PWrong = RwFlag2CalcWrongMLE(V,DTrain,c);
        nllhPWrongFolds(foldIdx, cIdx) = -mean(RwFlag2LogLik(DVali, dummyPi, PWrong));

        %--- Learn PFirst2, measure nllh
        PFirst2 = RwFlag2CalcFirst2MLE(V,DTrain,c);
        nllhPFirst2Folds(foldIdx, cIdx) = -mean(RwFlag2LogLik(DVali, dummyPi, PFirst2));

        %--- Learn Invite
        ObjParam.D = DTrain;
        ObjParam.c = c;
                
        p0 = RwFlag2GenP0(V); 
        beta0 = RwFlag2UncP2BetaMat(p0);
        
        lbfgsOpt = struct('maxIts', 4000, 'x0', beta0(mask), 'printEvery', 100);
        lb = -inf(size(lbfgsOpt.x0)); ub = inf(size(lbfgsOpt.x0));
        tic;
        [beta, fvalInviteTrain, outputInvite] = lbfgsb(@RwFlag2UncObj, lb, ub, lbfgsOpt);
        fprintf('lbfgsb function count = %d\n', outputInvite.totalIterations);
        toc
        
        PInvite = RwFlag2UncBetaVecToP(beta, V);
        nllhPInviteFolds(foldIdx, cIdx) = -mean(RwFlag2LogLik(DVali, dummyPi, PInvite));
      end
    end

    %--- choose the best c for each method
    nllhPWrongFolds
    me = mean(nllhPWrongFolds,1);
    [minIdx, tieList] = EvalFindBest(me, 'min');
    minIdx
    cWrongBest = cList(minIdx)
    cMat(tryIdx, mIdx, 2) = cWrongBest;

    nllhPFirst2Folds
    me = mean(nllhPFirst2Folds,1);
    [minIdx, tieList] = EvalFindBest(me, 'min');
    minIdx
    cFirst2Best = cList(minIdx)
    cMat(tryIdx, mIdx, 3) = cFirst2Best;

    nllhPInviteFolds
    me = mean(nllhPInviteFolds,1);
    [minIdx, tieList] = EvalFindBest(me, 'min');
    minIdx
    cInviteBest = cList(minIdx)
    cMat(tryIdx, mIdx, 1) = cInviteBest;

    %----- Learn estimators using all data.

    %--- Learn PWrong
    PWrong = RwFlag2CalcWrongMLE(V,DAll,cWrongBest)
    PCell{tryIdx, mIdx, 2} = PWrong;
    trainNllhMat(tryIdx, mIdx, 2) = -mean(RwFlag2LogLik(DAll, dummyPi, PWrong));
    fprintf('trainNllhMat(tryIdx, mIdx, 2) = '); disp(trainNllhMat(tryIdx, mIdx, 2));

    %--- Learn PFirst2
    PFirst2 = RwFlag2CalcFirst2MLE(V,DAll, cFirst2Best)
    PCell{tryIdx, mIdx, 3} = PFirst2;
    trainNllhMat(tryIdx, mIdx, 3) = -mean(RwFlag2LogLik(DAll, dummyPi, PFirst2));
    fprintf('trainNllhMat(tryIdx, mIdx, 3) = '); disp(trainNllhMat(tryIdx, mIdx, 3));

    %--- Learn Invite
    ObjParam.V = V;
    ObjParam.mask = mask;
    ObjParam.D = DAll;
    ObjParam.c = cInviteBest;

    p0 = RwFlag2GenP0(V); 
    lbfgsOpt = struct('maxIts', 4000, 'x0', beta0(mask), 'printEvery', 100);
    lb = -inf(size(lbfgsOpt.x0)); ub = inf(size(lbfgsOpt.x0));
    tic;
    [beta, fvalInviteTrain, outputInvite] = lbfgsb(@RwFlag2UncObj, lb, ub, lbfgsOpt);
    fprintf('lbfgsb function count = %d\n', outputInvite.totalIterations);
    toc

    PInvite = RwFlag2UncBetaVecToP(beta, V);
    PCell{tryIdx, mIdx, 1} = PInvite;
    trainNllhMat(tryIdx, mIdx, 1) = -mean(RwFlag2LogLik(DAll, dummyPi, PInvite));
    fprintf('trainNllhMat(tryIdx, mIdx, 1) = '); disp(trainNllhMat(tryIdx, mIdx, 1));
    
  end
end

fprintf('#--- Finished\n');
cMat
PCell
trainNllhMat

tmp = PCell{1,2,1}; norm(tmp(:) - modelP(:))
tmp = PCell{1,2,2}; norm(tmp(:) - modelP(:))
tmp = PCell{1,2,3}; norm(tmp(:) - modelP(:))

%--- To save
outFileName = 'result-batchunc.mat';
fprintf('Writing to %s ...\n', outFileName);
save(outFileName);

%--- key variables
%- trainNllhMat(tryIdx, mIdx, estIdx)
%- PCell{tryIdx, mIdx, estIdx}
%- cMat(tryIdx, mIdx, estIdx)
%- objMat(tryIdx, p0Idx, mIdx)
