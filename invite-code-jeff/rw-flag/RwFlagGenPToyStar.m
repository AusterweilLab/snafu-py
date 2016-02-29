function P = RwFlagGenPToyStar(V)
P = zeros(V,V);
P(1,:) = [0 ones(1,V-1)] / (V-1);
P(2:end,1) = 1;

end