function [basePathMap] = KjunReadMapFile(fNameBasePath)
% (DESCRIPTION)
%
% (IN)
%
% (OUT)
%
% (EX)
%
% $Author: deltakam $	$Date: 2014/12/03 12:04:23 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014

%- load
fp = fopen(fNameBasePath, 'r');
cols = textscan(fp, '%s %s', 'Delimiter', ',');
if (length(cols{1}) ~= 0)
  basePathMap = containers.Map(cols{1}, cols{2});
end
fclose(fp);

end
