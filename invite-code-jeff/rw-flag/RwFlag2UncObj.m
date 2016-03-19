function [ f, g ] = RwFlagUnc2Obj(x, varName)
globalVarName = 'ObjParam';
if (exist('varName','var'))
  globalVarName = varName;
end

eval(['global ' globalVarName]);
eval(['ObjParam = ' globalVarName ';']);
D = ObjParam.D;
c = ObjParam.c;
mask = ObjParam.mask;
V = ObjParam.V;
m = size(D,1);

beta = zeros(V,V);
beta(logical(diag(ones(1,V)))) = -inf;
beta(mask) = x;
expBeta = exp(beta);
P = bsxfun(@rdivide, expBeta, sum(expBeta,2));

%   %DEBUG
%   myEps = 1e-8;
%   betaDbg = zeros(V,V);
%   betaDbg(logical(diag(ones(1,V)))) = -inf;
%   betaDbg(mask) = x;
% 	betaDbg(3,2) = betaDbg(3,2) + myEps;
%   PDbg = bsxfun(@rdivide, exp(betaDbg), sum(exp(betaDbg),2));

nllh = 0; %negative log likelihood
if (nargout >= 2)
  grad = zeros(V,V);
end
for iter=1:m
  row = D(iter,:);
  
  len = find(row == 0,1,'first');
  if (isempty(len))
    len = length(row);
  else
    len = len - 1;
  end
  row = row(1:len);
  
  assert (len ~= 0 && length(unique(row(1:len))) == len);
  
  for k=1:len-1
    % compute the term for "from a_k to a_{k+1}
    nonasb = row(1:k);          % nonabsorbing
    asb = row(k+1:end);         % absorbing
    
    Q = P(nonasb,nonasb);
    R = P(nonasb,asb);
    
    %--- in case of numerical issues
    %       N = inv(eye(k) - Q);
    %       [L,U] = lu(eye(k) - Q);
    %       tmp = (U\(L\R(:,1)));
    %       prb = tmp(k,:);
    
%     dbstop if warning
    N = inv(eye(k) - Q);
%     dbclear if warning
    NR1 = N*R(:,1);
    prb = NR1(k);
    
    % compute negative log likelihood
    nllh = nllh - log(prb);
    
    %       %DEBUG
    %       prbDbg = compPrb();
    if (nargout >= 2)
      %- compute gradient
      QNR1 = Q*NR1;
      idx = [nonasb, asb];
      
      invIdx = zeros(1,V);
      invIdx(idx) = 1:len;
      
      for i=nonasb
        iInvIdx = invIdx(i);
        if (iInvIdx == 0 || iInvIdx > k)
          continue;
        end
        
        for j=1:V
          if (i == j)
            continue
          end
          
          v1 = -(QNR1(iInvIdx));
          if (invIdx(j) ~= 0 && invIdx(j) <= k)
            v1 = v1 + NR1(invIdx(j));
          end
          v1 = P(i,j)*v1;
          
          s = N(k,iInvIdx)*( v1 + ( -P(i,asb(1))*P(i,j) + (asb(1) == j)*P(i,asb(1)) ) );
          grad(i,j) = grad(i,j) - s/prb;
        end
      end
    end
    
    if (nargout >= 2)
      if (any(isnan(nllh) | isinf(nllh)) || any(isinf(grad(:)) | isnan(grad(:))))
        %         error('nan or inf found');
        %         keyboard;
        f = inf;
        grad = nan(V,V);
        g = grad(mask);
        return;
      end
    else
      if (any(isnan(nllh) | isinf(nllh)))
        f = inf;
        return;
      end
    end
    
  end
end

%--- regularizer
reg = .5*sum(x(:).^2);
regGrad = x;

if (c == 0) % incase of 0 * inf = nan case.
	f = nllh;
else
  f = nllh + c*reg;
end
if (nargout >=2)
  if (c == 0) % incase of 0 * inf = nan case.
    g = grad(mask);
  else
    g = grad(mask) + c*regGrad;
  end
end

%   function ret = compPrb()
%       QDbg = PDbg(nonasb,nonasb);
%       RDbg = PDbg(nonasb,asb);
%
%       NDbg = inv(eye(k) - QDbg);
%       ret = NDbg(k, :)*RDbg(:, 1);
%
%   end
end

