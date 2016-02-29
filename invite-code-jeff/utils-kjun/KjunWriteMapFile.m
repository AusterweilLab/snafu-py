function KjunWriteMapFile(basePathMap, fNameBasePath)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: deltakam $	$Date: 2014/12/03 12:05:50 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2014


fp = fopen(fNameBasePath, 'w');
keyList = basePathMap.keys();
for kIdx = 1:length(keyList)
  k = keyList{kIdx};
  fprintf(fp, '%s, %s\n', k, basePathMap(k));
end
fclose(fp);


end
