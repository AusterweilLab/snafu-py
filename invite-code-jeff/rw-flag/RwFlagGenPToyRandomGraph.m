function P = RwFlagGenPToyRandomGraph(V, p)
P = zeros(V,V);
for i=1:V-1
  for j=i+1:V
    if (rand() <= p)
      P(i,j) = 1;
      P(j,i) = 1;
    end
  end
end

P = bsxfun(@rdivide, P, sum(P,2));
P(isnan(P(:))) = 0;
  
end