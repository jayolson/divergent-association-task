import dat

# GloVe model from https://nlp.stanford.edu/projects/glove/
model = dat.Model("glove.840B.300d.txt", "words.txt")

# Compound words are translated into words found in the model
print(model.validate("cul de sac")) # cul-de-sac

# Compute the cosine distance between 2 words (0 to 2)
print(model.distance("cat", "dog")) # 0.1983
print(model.distance("cat", "thimble")) # 0.8787

# Compute the DAT score between 2 words (average cosine distance * 100)
print(model.dat(["cat", "dog"], 2)) # 19.83
print(model.dat(["cat", "thimble"], 2)) # 87.87

# Word examples (Figure 1 in paper)
low = ["arm", "eyes", "feet", "hand", "head", "leg", "body"]
average = ["bag", "bee", "burger", "feast", "office", "shoes", "tree"]
high = ["hippo", "jumper", "machinery", "prickle", "tickets", "tomato", "violin"]

# Compute the DAT score (transformed average cosine distance of first 7 valid words)
print(model.dat(low)) # 50
print(model.dat(average)) # 78
print(model.dat(high)) # 95
