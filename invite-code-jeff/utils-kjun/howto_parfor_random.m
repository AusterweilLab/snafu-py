seeds = randi(2^31-1,1,10, 'int32') 
parfor ii=1:10 
 rdstm = RandStream.create('mrg32k3a'); 
 reset(rdstm, seeds(ii)); 
 x(ii,:) = rdstm.randn(1,1); 
end 
