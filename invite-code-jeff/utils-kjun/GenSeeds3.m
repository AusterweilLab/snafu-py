function [seeds] = GenSeeds3(stream, m, n)
% (DESCRIPTION) 
% 
% (IN) 
%   stream: random stram. put [] to use the current global stream
% (OUT) 
% 
% (EX) 
% 
% $Author: kjun $	$Date: 2014/11/30 22:28:48 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

if (exist('m', 'var') && ~exist('n', 'var'))
  n = m;
  m = 1;
elseif (~exist('m','var'))
  m = 1;
  n = 1;
end
if (isempty(stream))
  stream = RandStream.getGlobalStream();
end
oldStream = RandStream.setGlobalStream(stream);
seeds = randi(2^31-1,m,n, 'int32');
RandStream.setGlobalStream(oldStream);

end
