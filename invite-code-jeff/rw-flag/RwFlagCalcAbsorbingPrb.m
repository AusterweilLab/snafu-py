function prbMat = RwFlagCalcAbsorbingPrb(P, nonabsorbing, absorbing)
% function prbMat = RwFlagCalcAbsorbingPrb(P, nonabsorbing)
% computes the probability of being absorbed to j when starting from j.
% IN  P: the transition matrix
%     nonabsorbing: a list of indices of P that corresponds to nonabsorbing states.
%     absorbing: a list of indices of P that corresponds to absorbing states.
% OUT prbMat: prbMat(i,j) is the proabability of being absorbed to
%             nonabsorbing(j) when the random walk is started from absorbing(i)

V = size(P,1);

asb = absorbing;
nonasb = nonabsorbing;

assert(length(unique([asb, nonasb])) == V);

Q = P(nonasb, nonasb);
R = P(nonasb, asb);

N = inv(eye(length(nonasb)) - Q);
prbMat = N*R;

end
