function P = RwFlagGenPToyGrid(V)
sqrtV = sqrt(V);
assert(sqrtV == floor(sqrtV));

P = zeros(V,V);
nodeNum = 1;

for i=1:sqrtV
  for j=1:sqrtV
    if (j~=1)
      P(nodeNum, nodeNum - 1) = 1;
    end
    if (j~=sqrtV)
      P(nodeNum, nodeNum + 1) = 1;
    end
    
    up = (nodeNum - sqrtV);
    down = (nodeNum + sqrtV);
    if (up >= 1)
      P(nodeNum, up) = 1;
    end
    if (down <= V)
      P(nodeNum, down) = 1;
    end
    
    P(nodeNum,:) = bsxfun(@rdivide, P(nodeNum,:), sum(P(nodeNum,:)));
    nodeNum = nodeNum + 1;
  end
end

end