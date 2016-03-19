function jsdVec = calcJsd(p1, p2)
%- p1 is either a prb vector or a matrice where each row is a prb vector.
%EX 
% >> calcJsd([.5 .5], [.6 .4])
% ans =
%     0.0051
% >> calcJsd([.5 .5;.5 .5], [.6 .4; .5 .5])
% ans =
%     0.0051
%     0.0051

[r,c] = size(p1);
if (r > 1 && c > 1)
  assert( all(p1(:)>=0) && all(abs(sum(p1,2) - 1)<1e-6));
  assert( all(p2(:)>=0) && all(abs(sum(p2,2) - 1)<1e-6));
else
  assert( all(p1(:)>=0) && all(abs(sum(p1)-1))<1e-6);
  assert( all(p2(:)>=0) && all(abs(sum(p2)-1))<1e-6);
  r = 1;
  p1 = reshape(p1, 1, length(p1));
  p2 = reshape(p2, 1, length(p2));
end
avgp = (p1+p2)/2;


jsdVec = zeros(r, 1);
for k=1:r
  jsd = 0;
  
  zeroIdx = avgp(k,:)==0;
  curavgp = avgp(k,~zeroIdx);
  curp1 = p1(k,~zeroIdx);
  curp2 = p2(k,~zeroIdx);
  
  v = 0;
  for i=1:length(curp1)
    if (curp1(i) ~= 0),
      v = v + curp1(i)*(log(curp1(i)) - log(curavgp(i)));
    end
  end
  jsd = jsd + .5*v;
  
  v = 0;
  for i=1:length(curp2)
    if (curp2(i) ~= 0),
      v = v + curp2(i)*(log(curp2(i)) - log(curavgp(i)));
    end
  end
  jsd = jsd + .5*v;
  
  jsdVec(k) = jsd;
end

end