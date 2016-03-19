function loglik = RwFlag2LogLik(D, pi,P)
% function loglik = RwFlagLogLik(pi,P,D)
% comoputes the log likelihood of the data given model parameters.
% IN D: 0-padded lists. D(i,j) is i-th list j-th item, which is 0 if the list is shorter than j.
%    pi: the initial distribution
%    P: the transition matrix
% OUT loglik: loglik(i) is the log likelihood of i-th list.

% warning('deprecate: (just the name of the function is misleading..')
loglik = -RwFlag2NllhPerList(P,D);

for i=1:size(D,1)
  loglik(i) = loglik(i) + log(pi(D(i,1)));
end
loglik = loglik';

end
