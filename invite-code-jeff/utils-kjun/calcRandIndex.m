function randIndex = calcRandIndex(z1,z2)

n = length(z1);
assert(n == length(z2));

a = 0; b = 0;
% c = 0; d = 0;
for i=1:n-1
  for j=i+1:n
    sameInZ1 = (z1(i) == z1(j));
    sameInZ2 = (z2(i) == z2(j));
    
    if (sameInZ1 && sameInZ2)
      a = a + 1;
    end
    if (~sameInZ1 && ~sameInZ2)
      b = b + 1;
    end
%     if (sameInZ1 && ~sameInZ2)
%       c = c + 1;
%     end
%     if (~sameInZ1 && sameInZ2)
%       d = d + 1;
%     end
    
  end
end

randIndex = (a+b)/nchoosek(n,2);
end