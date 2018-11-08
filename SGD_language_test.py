import SGD_classifier as SGD

# Data setup
en = [line.strip() for line in open('en.txt') if line.strip() != '']
de = [line.strip() for line in open('de.txt') if line.strip() != '']
nl = [line.strip() for line in open('nl.txt') if line.strip() != '']
af = [line.strip() for line in open('af.txt') if line.strip() != '']
sv = [line.strip() for line in open('sv.txt') if line.strip() != '']
entest = [line.strip() for line in open('entest.txt') if line.strip() != '']
detest = [line.strip() for line in open('detest.txt') if line.strip() != '']
nltest = [line.strip() for line in open('nltest.txt') if line.strip() != '']
aftest = [line.strip() for line in open('aftest.txt') if line.strip() != '']
svtest = [line.strip() for line in open('svtest.txt') if line.strip() != '']


# Feature engineering
def lfeaturize(line):
    featvec = {a+b for a,b in list(zip(line,line[1:]))}
    return(featvec)

trainfeatures = []
trainclasses = []
for line in en[:400]:
    trainfeatures.append(lfeaturize(line))
    trainclasses.append("EN")
for line in de[:400]:
    trainfeatures.append(lfeaturize(line))
    trainclasses.append("DE")
for line in nl[:400]:
    trainfeatures.append(lfeaturize(line))
    trainclasses.append("NL")
for line in af[:400]:
    trainfeatures.append(lfeaturize(line))
    trainclasses.append("AF")
for line in sv[:400]:
    trainfeatures.append(lfeaturize(line))
    trainclasses.append("SV")

devfeats = []
devclasses = []
for line in en[400:420]:
    devfeats.append(lfeaturize(line))
    devclasses.append("EN")
for line in de[400:420]:
    devfeats.append(lfeaturize(line))
    devclasses.append("DE")
for line in nl[400:420]:
    devfeats.append(lfeaturize(line))
    devclasses.append("NL")
for line in af[400:420]:
    devfeats.append(lfeaturize(line))
    devclasses.append("AF")
for line in sv[400:420]:
    devfeats.append(lfeaturize(line))
    devclasses.append("SV")

classifier = SGD.SGDClassifier(lossfn="perceptron", shuffle=True, max_iter=50, verbose=True)
classifier.fit(trainfeatures, trainclasses)
classified = classifier.predict(trainfeatures)

#for item in range(len(classified)):
    #print(classified[item], trainclasses[item])

cor = 0
inc = 0

for item in range(len(classified)):
    if classified[item] == trainclasses[item]:
        cor += 1
    else:
        inc += 1

print("CORRECT: ", cor)
print("INCORRECT: ", inc)
print("ACCURACY: ", cor/(cor+inc))