global INITSCRIPT_TIME INITSCRIPT_EXISTFNAMEDIARY INITSCRIPT_FNAMEDIARY
fprintf('%s\n', repmat('#',1,80));
fprintf('# Ending time: '); disp(datestr(now()))
fprintf('# ');toc(INITSCRIPT_TIME);
if (INITSCRIPT_EXISTFNAMEDIARY)
  fprintf('# Diary file: %s\n', INITSCRIPT_FNAMEDIARY);
end
fprintf('%s\n', repmat('#',1,80));

if (INITSCRIPT_EXISTFNAMEDIARY)
  diary off;
end
