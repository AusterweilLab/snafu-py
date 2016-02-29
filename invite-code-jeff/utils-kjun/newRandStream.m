function stream=newRandStream(seed)
% function stream=newRandStream(seed)
% DESCRIPTION 
%  creates a new random stream based on 'seed'
% INTPUT
%  seed: seed number for generating a new random stream
% OUTPUT
%  Newly created random stream
% Example
%  defaultStream = getRandStream();
%  setRandStream(newRandStream(seed));
%  % in here, whenever you call rand() or randn(), etc. , you are using a new stream based on 'seed'
%  setRandStream(defaultStream); % setup the stream that's been used before.
stream = RandStream('mt19937ar','Seed',seed);
