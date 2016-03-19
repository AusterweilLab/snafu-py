function [ b,m ] = countUnique( A )
% function [ b,m ] = countUnique( A )
% only works for rows. b is sorted unique members of A. m is occurrences
% ex.
% C = [2     3;
%      1     2;
%      3     3;
%      2     3;
%      1     2;
%      2     2;
%      3     1;
%      3     1];
% [b,m]=countUnique(C)
% b =
%      1     2
%      2     2
%      2     3
%      3     1
%      3     3
% m =
%      2
%      1
%      2
%      2
%      1


A=sortrows(A);
[b,m] = unique(A,'rows', 'legacy');
m = [m(1);diff(m)];


end

