function opt = FMinUncASgdOptions()
  opt = struct();
  opt.stepsizeGamma0 = .1;
  opt.stepsizeA = 1; % 0.001 recommended
  opt.stepsizeC = 2/3;
  opt.objC = 1e-4; % regularization constant. small: less regularized. 
  opt.nEpoch = 5;
  opt.shuffle = true;
  opt.dbg = false;
  opt.dbgData = zeros(0,0);
  opt.dbgInterval = -1;
  opt.dbgMeasureTrainObj = true;
end
