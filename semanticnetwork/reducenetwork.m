% take the big semantic network and reduce it to only animals

load tempMar25.mat;

animalnet = zeros(length(aniWords));

[i,j]=find(linkMatrix);

[y1, i1]=ismember(words(i),aniWords);
[y2, i2]=ismember(words(j),aniWords);

for k=1:length(i)
    if ((i1(k) > 0) & (i2(k) > 0))
        animalnet(i1(k),i2(k))=1;
    end
    fprintf('%i\n', k)
end

save('animalnet.mat', 'animalnet','aniWords');
csvwrite('animalnet.csv',animalnet);

fid=fopen('animalwords.csv','w');
fprintf(fid,'%s\n',aniWords{:});
fclose(fid);
