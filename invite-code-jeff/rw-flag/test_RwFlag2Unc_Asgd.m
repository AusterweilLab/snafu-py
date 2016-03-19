initScript;

%------- make sure the derivative is correct.
rng(0);

load('RwFlagToy.mat');
% dset = 
%           D: [1000x6 double]
%      modelP: [6x6 double]
%     modelPi: [0.1667 0.1667 0.1667 0.1667 0.1667 0.1667]
POrig = dset.modelP;
[m,V] = size(dset.D);
c = 1e-4;
nEpoch = 10;
mask = RwFlag2UncMask(V);

mTrain = 100;
DTrain = dset.D(1:mTrain,:);
DVali = dset.D(mTrain+(1:mTrain),:);

%---- count to find the MLE.
PFirst2 = RwFlag2CalcFirst2MLE(V,DTrain,c);
PWrong = RwFlag2CalcWrongMLE(V,DTrain,c)

tmp = PFirst2; tmp = tmp + 1e-8; % smoothing
tmp(logical(eye(V))) = 0;
tmp = bsxfun(@rdivide, tmp, sum(tmp,2));
betaFirst2 = RwFlag2UncP2BetaMat(tmp);
% global ObjParam;
% ObjParam.V= V;
% ObjParam.D = DTrain;
% ObjParam.c = c;
% ObjParam.mask = mask;
% [fFirst2,~] = RwFlag2UncObj(betaFirst2(mask));

%---- prepare option struct
opt = FMinUncAsgdOptions();
opt.nEpoch = nEpoch;
opt.objC = c;
opt.dbg = true;
opt.dbgData = DVali;
opt.dbgInterval = 20;
opt.stepsizeGamma0 = 1;% .1 sucked
opt.stepsizeA = 0.001;
opt.stepsizeC = 3/4;% 2/3;
opt.mask = mask;
opt

%--- TODO could use the count estimate to start the optimization
%- but for debugging purpose, I will not use it now.
p0 = RwFlag2GenP0(V);
beta0 = RwFlag2UncP2BetaMat(p0);
% beta0 = betaFirst2;
tic;
[beta,output] = FMinUncAsgd(@RwFlag2UncObj, beta0(mask), V, DTrain, opt);
toc
output

beta = RwFlag2UncBetaVecToMat(beta, V);
PSol = bsxfun(@rdivide, exp(beta), sum(exp(beta),2));

%----- find the true MLE sol (mle-gd)
global ObjParam
ObjParam.V = V;
ObjParam.D = DTrain;
ObjParam.c = c;
ObjParam.mask = mask;


lbfgsOpt = struct('maxIts', 4000, 'x0', beta0(mask), 'printEvery', 100);
lb = -inf(size(lbfgsOpt.x0)); ub = inf(size(lbfgsOpt.x0));
tic;
[beta, fvalBatchTrain, outputBatch] = lbfgsb(@RwFlag2UncObj, lb, ub, lbfgsOpt);
fprintf('lbfgsb function count = %d\n', outputBatch.totalIterations);
toc
% ObjParam.D = DVali;
% ObjParam.c = 0;
% nllhBatchVali = RwFlag2UncObj(beta);

beta = RwFlag2UncBetaVecToMat(beta, V);
PSolBatch = bsxfun(@rdivide, exp(beta), sum(exp(beta),2));
nllhBatchVali = sum(RwFlag2NllhPerList(PSolBatch, DVali));




% fminuncopt = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off', 'GradObj', 'on');
% tic;
% [beta,fvalBatchTrain,exitflagBatch,outputBatch,~,~] = ...
%   fminunc(@RwFlag2UncObj, betaFirst2(mask), fminuncopt);
% fprintf('fminunc function count = %d\n', outputBatch.funcCount);
% toc
% ObjParam.D = DVali;
% fvalBatchVali = RwFlag2UncObj(beta);
% beta = RwFlag2UncBetaVecToMat(beta, V);
% PSolBatch = bsxfun(@rdivide, exp(beta), sum(exp(beta),2));

PSol
PSolBatch
PFirst2
POrig 
PWrong

fprintf('Error from the mle-sgd:\n');
sum(abs(PSol(:)-POrig(:)).^2)
fprintf('Error from the mle-gd:\n');
sum(abs(PSolBatch(:)-POrig(:)).^2)
fprintf('Error from PFirst2:\n');
sum(abs(PFirst2(:)-POrig(:)).^2)
fprintf('Error from PWrong:\n');
sum(abs(PWrong(:)-POrig(:)).^2)

t = output.dbgObjTimeList;
epoch = t/mTrain;
objTrain = output.dbgObjTrain;
nllhVali = output.dbgNllhVali;
figure; plot(epoch,objTrain); hold on; 
plot(epoch, nllhVali,'r'); 
plot(epoch, fvalBatchTrain*ones(1,length(epoch)), '-.');
plot(epoch, nllhBatchVali*ones(1,length(epoch)), 'r-.');
legend({'sgd, train','sgd, validation','gd, train', 'gd, validation'});

