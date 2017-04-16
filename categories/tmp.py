# troyer-hills animals
animals=[]
with open('troyer_hills_animals.csv','r') as th:
    for line in th:
        animal = line.split(',')[1].strip('\n').lower().replace(" ","")
        if animal not in animals:
            animals.append(animal)

newanimallist=[]
with open('results_unclean.csv','r') as newanimals:
    for line in newanimals:
        animal = line.split(',')[3]
        if (animal not in animals) and (animal not in newanimallist):
            print animal
            newanimallist.append(animal)
            
