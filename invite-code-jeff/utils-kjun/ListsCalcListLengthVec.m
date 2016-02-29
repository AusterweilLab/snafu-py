function lenVec = ListsCalcListLengthVec(mat)
%- compute zero-padded list length.
%- TEST
% ListsCalcListLength([1,2,3,0])
% ans =
%      3

m=size(mat,1);
lenVec = zeros(m,1);

for i=1:m
  lenVec(i) = ListsCalcListLength(mat(i,:));
end


end