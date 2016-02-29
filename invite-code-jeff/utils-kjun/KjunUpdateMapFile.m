function KjunUpdateMapFile(fNameBasePath, hostname, curPathFull)
% (DESCRIPTION) 
% 
% (IN) 
% 
% (OUT) 
% 
% (EX) 
% 
% $Author: deltakam $	$Date: 2015/05/04 16:35:31 $	$Revision: 0.1 $
% Copyright: Kwang-Sung Jun 2015

%- lock
fNameLock = '.kjunbasepath.lock.d';
if (ispc == false)
%   system(sprintf('lockfile %s </dev/null', fNameLock));
  ret = system(sprintf('mkdir %s </dev/null', fNameLock));
  if (ret == 1)
    while (ret == 1) % while failing
      fprintf('Locked. Sleeping 2 secs...\n');
      system('sleep 2 </dev/null');
      ret = system(sprintf('mkdir %s </dev/null', fNameLock));
    end
  end
end

%- read
existFile = exist(fNameBasePath, 'file');
basePathMap = containers.Map();
if (existFile)
  basePathMap = KjunReadMapFile(fNameBasePath);
end

%- update
basePathMap(hostname) = curPathFull;

%- write out
KjunWriteMapFile(basePathMap, fNameBasePath);

%- unlock
if (ispc == false)
%   system(sprintf('rm -f %s </dev/null', fNameLock));
  ret = system(sprintf('rmdir %s </dev/null', fNameLock));
  assert(ret == 0);
end

end
