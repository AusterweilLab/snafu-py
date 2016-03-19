function [dev, threshold, pval] =  getDeviation(alfa, diffAry)

n = length(diffAry);
threshold = tinv(1-alfa/2, n-1);
me = mean(diffAry);
dev = me / sqrt(sum((diffAry - me).^2)/(n*(n-1)));

x = 0.05;
lb = 0;
ub = 1;
while true
  thres = tinv(1-x/2, n-1);
  if (abs(dev) > thres) %overshooted
    ub = x;
  else
    lb = x;
  end
  x = (lb + ub) / 2;
  
  if (ub - lb) < 1e-6
    break;
  end
end
pval = ub;



end