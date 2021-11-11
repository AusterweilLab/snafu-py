"""
1. Download the file from here https://github.com/commonsense/conceptnet5/wiki/Downloads
2. Install pandas (pip install pandas)
3. Install nltk (pip install nltk)
4. Install python-Levenshtein (pip install python-Levenshtein)
5. (optional) Install tqdm (pip install tqdm) for progress bars
"""

import pandas, nltk, Levenshtein, tqdm

CONCEPTNET_FILE_PATH = "/raid/datasets/conceptnet/conceptnet-assertions-5.7.0.csv" # use your own path

# USING CONCEPTNET TO CORRECT
# conceptnet = pandas.read_csv(CONCEPTNET_FILE_PATH, sep="\t", names=["edge", "relation", "start-node", "end-node", "extra-info"]).drop("edge", 1).drop("extra-info", 1)
# conceptnet = conceptnet[conceptnet["relation"].isin(["/r/AtLocation", "/r/DefinedAs", "/r/FormOf", "/r/InstanceOf", "/r/IsA", "/r/MannerOf", "/r/PartOf"])]
# conceptnet = conceptnet[conceptnet["start-node"].str.startswith("/c/en") & conceptnet["end-node"].str.startswith("/c/en")]
# conceptnet["relation"] = conceptnet["relation"].apply(lambda x: x.split("/")[2])
# conceptnet["start-node"] = conceptnet["start-node"].apply(lambda x: x.split("/")[3])
# conceptnet["end-node"] = conceptnet["end-node"].apply(lambda x: x.split("/")[3])

allCategories = ["foods", "tools", "supermarket items", "vegetables", "animals", "fruits"]
allWords = {category: set() for category in allCategories}
lemmatizer = nltk.stem.WordNetLemmatizer()

# # level 1
# for category in allCategories:
#     for i, row in conceptnet[conceptnet["start-node"].str.contains(lemmatizer.lemmatize(category))].iterrows():
#         allWords[category].add(row["start-node"])
#         allWords[category].add(row["end-node"])
#     for i, row in conceptnet[conceptnet["end-node"].str.contains(lemmatizer.lemmatize(category))].iterrows():
#         allWords[category].add(row["start-node"])
#         allWords[category].add(row["end-node"])
# with open("temp/allWords-level1", "w+") as f:
#     f.write(str(allWords))

# # level 2
# for category in allCategories:
#     for item in list(allWords[category]):
#         for i, row in conceptnet[conceptnet["start-node"].str.contains(item)].iterrows():
#             allWords[category].add(row["start-node"])
#             allWords[category].add(row["end-node"])
#         for i, row in conceptnet[conceptnet["end-node"].str.contains(item)].iterrows():
#             allWords[category].add(row["start-node"])
#             allWords[category].add(row["end-node"])
# with open("temp/allWords-level2", "w+") as f:
#     f.write(str(allWords))

# CREATING ANOTHER VERSION OF SNAFU_SAMPLE
with open("temp/allWords-level2", "r") as f:
    allWords = eval(f.readline())
fo = open("snafu_sample_cleaned.csv", "w+")
fint = open("temp/changes.txt", "w+")
typos = set()
with open("../fluency_data/snafu_sample.csv", "r") as f:
    fo.write(f.readline())
    for line in tqdm.tqdm(f.readlines()):
        line = line.split(",")
        minLevenshtein, replaceWith = len(line[3]), line[3]
        if line[3] not in allWords[line[2]]:
            for word in allWords[line[2]]:
                l = Levenshtein.distance(line[3], word)
                if l < minLevenshtein:
                    replaceWith = word
                    minLevenshtein = l
        new_word = lemmatizer.lemmatize(replaceWith).replace("_", " ")
        if new_word != line[3]:
            if line[3] not in typos:
                fint.write(line[3] + "\t" + new_word + "\n")
                typos.add(line[3])
            line[3] = new_word
        fo.write(",".join(line))

fo.close()
fint.close()