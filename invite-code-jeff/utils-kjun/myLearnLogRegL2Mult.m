function wCell = myLearnLogRegL2Mult(X,y,nClasses, lambdaList, wInit, mfOptions)
% INPUT
% mfOptions.Method = 'lbfgs'; % or 'newton'
% mfOptions.Display = 0; %'final' or 'full'
% wCell{k} : nVars+1 by nClasses-1

%%% Logistic Regression
X = [ones(size(X,1),1) X]; % Add Bias element to features
[nInstances, nVars] = size(X);

funObj = @(w)SoftmaxLoss2(w,X,y,nClasses);
if (isempty(wInit))
  % Initialize Weights and Objective Function
  wInit = zeros(nVars,nClasses-1);
end
wInit = wInit(:);


wCell = cell(1,length(lambdaList));
% wMat = zeros(length(lambdaList), nVars+1);
prevW = wInit;
for i=1:length(lambdaList)
  lambda = lambdaList(i)*ones(nVars,nClasses-1);
  lambda(1,:) = 0; % Don't regularize bias elements
  lambda = lambda(:);
  
  funObjL2 = @(w)penalizedL2(w,funObj,lambda);
  W = minFunc(funObjL2,prevW,mfOptions);
  
  prevW = W;
  wCell{i} = reshape(W,nVars,nClasses-1);
end

end