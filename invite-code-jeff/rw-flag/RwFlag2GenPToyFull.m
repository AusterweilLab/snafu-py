function [P] = RwFlag2GenPToyFull(V)
% DESCRIPTION 
%   Generates a full graph
% IN 
%   V: # of nodes
% OUT 
% 
% EX 
% 
% $Author: deltakam $	$Date: 2014/10/02 16:08:00 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

P = ones(V,V);
P(logical(eye(V))) = 0;

P = bsxfun(@rdivide, P, sum(P,2));

end
