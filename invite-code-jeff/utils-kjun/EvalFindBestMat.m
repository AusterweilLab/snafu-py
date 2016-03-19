function [bestIdx, tieIdxMat] = EvalFindBestMat(mat, direction)
% DESCRIPTION 
%   Note that the 'best' among tie is somewhat arbitrary.
% IN 
% 
% OUT 
% 
% EX 
%   [bestIdx, tieIdxMat] = EvalFindBestMat([1,2,4;3,4,1], 'max')
%   bestIdx =
%        2     2
%   tieIdxMat =
%        2     2
%        1     3
% $Author: deltakam $	$Date: 2014/10/16 14:45:24 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

assert(strcmp(direction, 'min') || strcmp(direction, 'max'));

sz = size(mat);
[bestIdx, tieList] = EvalFindBest(mat(:), direction);

[r,c]=ind2sub(sz, bestIdx);
bestIdx = [r,c];

tieIdxMat = zeros(length(tieList), 2);
for i=1:length(tieList)
  [r,c]=ind2sub(sz, tieList(i));
  tieIdxMat(i,:) = [r,c];
end

end
