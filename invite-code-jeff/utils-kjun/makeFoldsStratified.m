function [ folds ] = makeFoldsStratified(n, K, y)
% function [ folds ] = makeFoldsStratified(n, K, y)
% DESCRIPTION make K folds out of n data (assuming index is from 1 to n) while keeping class
% proportion the same (as much as possible)
% INPUT
% n : # data points
% K : # folds
% y : class values of the data. (this is required because of 'starification' that keeps
%     class proportion same for each fold.
% OUTPUT
% folds : cell array of length K. Each fold, folds{i}, contains indices of data points
% that belong to the i'th fold.

% keeps initial index for later use !
y = reshape(y, length(y),1);
y = [(1:n)' y];
% origy = y;

idx = randperm(n);
y=y(idx,:);

[~,idx]=sort(y(:,2));
y=y(idx,:);


folds = cell(K,1);
for k=1:K
  folds{k} = y(k:K:n,1); % only take the index
end


end
