function dbgprint(varargin)
global KJUN_DEBUG
if (~isempty(KJUN_DEBUG) && KJUN_DEBUG == true)
  fprintf(varargin{:});
end
end