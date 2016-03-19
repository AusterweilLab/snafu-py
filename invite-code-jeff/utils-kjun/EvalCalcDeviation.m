function dev =  EvalCalcDeviation(ary, alfa)
%- ary(i,j) = i'th method, j'th trial

if (~exist('alfa', 'var'))
  alfa = 0.05;
end

n = length(ary);
stdErr = std(ary,0,2)/sqrt(n);
dev = stdErr * tinv(1-alfa/2, n-1);

end
