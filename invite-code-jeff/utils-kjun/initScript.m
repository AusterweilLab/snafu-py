function initScript(bClear, fNameDiary)
% - bClear does not matter
if (~exist('bClear', 'var'))
  bClear = true;
end
if (bClear)
  evalin('caller','clear all');
end

global INITSCRIPT_EXISTFNAMEDIARY INITSCRIPT_FNAMEDIARY
INITSCRIPT_EXISTFNAMEDIARY = exist('fNameDiary', 'var');
if (INITSCRIPT_EXISTFNAMEDIARY)
  diary(fNameDiary);
  INITSCRIPT_FNAMEDIARY = fNameDiary;
end

evalin('caller', 'dbstop if error');
evalin('caller', 'format compact');
evalin('caller', 'format short');
fprintf('%s\n', repmat('#',1,80));
fprintf('# Starting time: '); disp(datestr(now()))
if (INITSCRIPT_EXISTFNAMEDIARY)
  fprintf('# Diary file: %s\n', INITSCRIPT_FNAMEDIARY);
end
fprintf('%s\n', repmat('#',1,80));

global INITSCRIPT_TIME
INITSCRIPT_TIME = tic();
end