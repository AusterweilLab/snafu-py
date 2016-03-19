function oldStream = setRandStream(newStream)
% function setRandStream(newStream)
% DESCRIPTION 
%  sets current stream as 'newStream' argument
% INTPUT
%  newStream: RandStream object
% OUTPUT
%  Returns current default stream
% Example
%  defaultStream = setRandStream(newRandStream(seed));
%  % in here, whenever you call rand() or randn(), etc. , you are using a new stream based on 'seed'
%  setRandStream(defaultStream); % setup the stream that's been used before.
oldStream = RandStream.getGlobalStream();
RandStream.setGlobalStream(newStream);
