with open('group_2015.csv','r') as fi:
    with open('new.csv','w') as fo:
        header=fi.readline()
        fo.write(header)
        subj=0
        game=0
        newgame=0
        for line in fi:
            line=line.split(',')
            if (subj != line[0]) or (game != line[1]):
                subj=line[0]
                game=line[1]
                newgame += 1
            line[0]='S101'
            line[1]=str(newgame)
            fo.write(",".join(line))
