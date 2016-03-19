function wMat = myLearnLogRegL2(X,y,lambdaList, wInit, method, verbose)
% INPUT
%  X: nInstances by nVars
%  y: nVars by 1
%  lambdaList: arbitrary length of lambda's. However, decreasing order is suggested for the best speed
%              large lambda means large penalization for the coefficients.
%  verbose: 0: nothing, 1: only termination condition. 2 for everything.
% OUTPUT
%  wMat: length(lambdaList) by (nVars+1)


[nInstances, nVars] = size(X);
%%% Logistic Regression
X = [ones(nInstances,1) X]; % Add Bias element to features

funObj = @(w)LogisticLoss(w,X,y);
if (isempty(wInit))
  wInit = zeros(1,nVars+1);
end

if (~exist('method','var'))
  method = 'newton';
end
  
mfOptions.Method = method;
% display. 0: nothing, 'final': only termination condition. 'full' for everything.
switch verbose
  case 0
    mfOptions.Display = 0;  
  case 1
    mfOptions.Display = 'final';  
  case 2
    mfOptions.Display = 'full';  
  otherwise
    error('invalid verbose value');
end

wMat = zeros(length(lambdaList), nVars+1);
prevW = wInit;
for i=1:length(lambdaList)
  lambda = lambdaList(i)*ones(nVars+1,1);
  lambda(1) = 0; % Do not penalize bias variable
  funObjL2 = @(w)penalizedL2(w,funObj,lambda);

  wMat(i,:) = minFunc(funObjL2, prevW', mfOptions);
  prevW = wMat(i,:);
end

end