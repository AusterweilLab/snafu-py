function P = RwFlagGenPToyCircle(V, p)
P = zeros(V,V);
for i=1:V
  if (i == V), ip1 = 1; else ip1 = i + 1; end
  P(i,ip1) = p;
  P(ip1,i) = (1-p);
end

end