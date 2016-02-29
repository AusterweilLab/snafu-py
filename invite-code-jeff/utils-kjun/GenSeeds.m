function seeds = GenSeeds(nSeeds, seed0)
if (exist('seed0','var'))
  oldState = rng(seed0);
end
seeds = randi(2^31-1,1,nSeeds, 'int32');
if (exist('seed0','var'))
  rng(oldState);
end

end