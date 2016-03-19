function [ trainIdxList ] = CVGetTrainSet( folds, k )
% function [ trainIdxList ] = CVGetTrainSet( folds, k )

nData = sum(cellfun(@length, folds));
trainIdxList = setdiff(1:nData, folds{k});

end

