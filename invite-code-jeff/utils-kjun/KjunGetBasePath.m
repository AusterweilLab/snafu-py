function [ret] = KjunGetBasePath()
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
%    The path to the /code folder of the repository
% (EX) 
% 
% $Author: deltakam $	$Date: 2014/10/28 13:23:41 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

fNameBasePath = 'KjunBasePath.txt';
hostname = getComputerName();

basePathMap = KjunReadMapFile(fNameBasePath);

if (~basePathMap.isKey(hostname))
  error('hostname can''t be found. Did you run addKjunPaths.m?');
end

ret = basePathMap(hostname);
end
