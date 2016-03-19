function [x,output] = FMinUncAsgd(obj, x0, V, D, opt)
if (~exist('opt','var'))
  opt = FMinUncAsgdOptions();
end
ssgamma0 = opt.stepsizeGamma0;
ssa = opt.stepsizeA;
ssc = opt.stepsizeC;

% load data from the global variables.
global ObjParam ObjParamDbgTrain ObjParamDbgVali
% eval(sprintf('global %s ObjParam', opt.DataVarName));
% eval(sprintf('data = %s', opt.DataVarName));
m = size(D,1);
mask = opt.mask;

% prepare loop.
fList = nan(m, opt.nEpoch);
stepNormList = nan(m, opt.nEpoch);% ??
tt = 1:(opt.nEpoch*m);
stepsize = ssgamma0*((1 + ssa*ssgamma0*tt).^(-ssc));

ObjParam.V = V;
ObjParam.c = opt.objC / m; % PER list regularization
ObjParam.mask = mask;
if (opt.dbg)
  ObjParamDbgTrain = ObjParam;
  ObjParamDbgTrain.c = opt.objC; % WHOLE list regularization
  ObjParamDbgTrain.D = D;
  ObjParamDbgVali = ObjParam;
%   if (opt.bMeasureNllh)
%     ObjParamDbgVali.c = 0;         % compute THE log likelihood (leaves out regularization)
%   else
%     ObjParamDbgVali.c = opt.objC;  % WHOLE list regularization
%   end
  ObjParamDbgVali.c = 0;         % compute THE log likelihood (leaves out regularization)
  ObjParamDbgVali.D = opt.dbgData;
  dbgObjTrain = [];
  dbgNllhVali = [];
  dbgObjTimeList = [];
end
t = 1;
x = x0;
barx = zeros(size(x));
for iter = 1:opt.nEpoch
  % sort
  idxList = randperm(m);
  
  for ii=1:m
    ObjParam.D = D(idxList(ii),:);
    [f,g] = obj(x);
    
    step = stepsize(t)*g;
    x = x - step;
    
    if (any(isnan(x)) || any(isinf(x)))
      x = nan(size(x));
      output.finalx = [];
      output.finalbarx = [];
      output.stepNormList = [];
      output.msg = 'optimization failed by having nan or inf x';
      output.exitflag = 1;
      return;
    end
    
    barx = ((t-1)*barx + x)/t; % this is our solution
    fList(iter,ii) = f;
    stepNormList(ii,iter) = norm(step);

    %- for debugging: check true train/test objective
    if (opt.dbg && (t == 1 || mod(t,opt.dbgInterval) == 0) )
      %- compute train obj, test obj
      if (opt.dbgMeasureTrainObj)
        [f] = obj(barx, 'ObjParamDbgTrain');
        dbgObjTrain(end+1) = f;
      end
      [f] = obj(barx, 'ObjParamDbgVali');
      dbgNllhVali(end+1) = f;
      
      dbgObjTimeList(end+1) = t;
    end
    
    t = t + 1;
  end
end

%put info into output variable.
output.finalx    = x;
output.finalbarx = barx;
output.stepNormList = stepNormList;
output.msg = 'no error';
output.exitflag = 0;
if (opt.dbg)
  output.dbgObjTimeList = dbgObjTimeList;
  output.dbgObjTrain = dbgObjTrain;
  output.dbgNllhVali = dbgNllhVali;
end

x = barx;
end








