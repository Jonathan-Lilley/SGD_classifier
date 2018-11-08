import numpy as np
from random import randint, shuffle
from math import exp,log


class SGDClassifier:

    def __init__(self, lossfn = "perceptron", lr = 0.1, max_iter = 20, shuffle = True, verbose = False):
        self.lossfn = lossfn.lower() # The loss function specified
        self.lr = lr # Learning rate
        self.max_iter = max_iter # The maximum iterations through the main loop
        self.shuffle = shuffle # Shuffle training examples
        self.verbose = verbose # Outputs runtime data
        self.trained = False # Checks if trained
            
    # Activation function according to loss function
    def activation(self,i,outs):
        # Perceptron activation function
        if self.lossfn == "perceptron":
            if outs[i] > 0:
                return 1
            return 0
        # Hinge loss activation function
        elif self.lossfn == "hinge":
            if outs[i] > 0:
                return 1
            return -1
        # Sigmoid/logistic activation
        elif self.lossfn == "logistic":
            return(exp(outs[i]-max(outs))/sum([exp(j - max(outs)) for j in outs]))
    
    # Classifies outputs s.t. the highest output is the guessed output
    # Returns guessed output and scores of inputs after activation fn
    # Scores are returned for use in hinge loss optimization
    def classify(self, outs):
        scores = {i:self.activation(i,outs) for i in range(len(outs))}
        return max(scores, key=scores.get), scores
    
    # Defines optimization based on loss function
    def update(self,f,c,g,gval,scores):
        # Perceptron optimization
        if self.lossfn == "perceptron":
            if c != g:
                for feat in f:
                    if feat in self.inttofeat:
                        self.weights[c][feat] += 1
                        self.weights[g][feat] -= 1
        # Logistic regression optimization
        elif self.lossfn == "logistic":
            if c != g:
                for feat in f:
                    if feat in self.inttofeat:
                        self.weights[c][feat] += gval*feat*self.lr
                        self.weights[g][feat] -= gval*feat*self.lr
            else:
                for feat in f:
                    if feat in self.inttofeat:
                        self.weights[c][feat] += (gval-1)*feat*self.lr
                        self.weights[g][feat] -= (gval-1)*feat*self.lr
        # Hinge loss optimization
        elif self.lossfn == "hinge":
            for feat in f:
                if feat in self.inttofeat:
                    if c!= g:
                        self.weights[c][feat] += 1
                    for l in range(len(self.weights)):
                        if scores[l] == 1 and l != c:
                            self.weights[l][feat] -= 1
    
            
    def fit(self, features, classes, devfeatures = [], devclasses = []):
        """Trains SGD classifier given features and corresponding classes."""
        
        # Converts all features to strings to make them sortable
        features = [[str(f) for f in feats] for feats in features]
        devfeatures = [[str(f) for f in feats] for feats in devfeatures]
        
        # Map features to integers starting from 0
        self.num_examples = len(features)
        self.num_devexamples = len(devfeatures)
        fset = sorted(list(set([f for g in features + devfeatures for f in g])))
        self.inttofeat = dict(zip(range(len(fset)), fset))
        self.feattoint = dict(zip(fset, range(len(fset))))
        self.features = [[self.feattoint[f] for f in g] for g in features]
        self.num_features = len(fset)
        self.devfeatures = [[self.feattoint[f] for f in g] for g in devfeatures]
        
        # Map classes to integers starting from 0        
        cset = sorted(list(set([c for c in classes + devclasses])))
        self.inttoclass = dict(zip(range(len(cset)), cset))
        self.classtoint = dict(zip(cset, range(len(cset))))
        self.classes = [self.classtoint[f] for f in classes]
        self.devclasses = [self.classtoint[f] for f in devclasses]
        self.num_classes = len(cset)
        
        # Creates weight vectors
        self.weights = [[randint(-1,1) for f in range(self.num_features)] for c in range(self.num_classes)]
        
        # Consolidates and shuffles data (if shuffle set to true)
        examples = list(zip(self.classes,self.features))
        devexamples = list(zip(self.devclasses,self.devfeatures))
        if shuffle == True:
            examples = shuffle(examples)
            devexamples = shuffle(devexamples)
        
        # Main loop
        prevdeverr = 0
        for iteration in range(self.max_iter):
            err,deverr = 0,0
            for c,f in examples:
                outs = [sum([w[feat] for feat in f]) for w in self.weights]
                guess, scores = self.classify(outs)
                if self.lossfn == "hinge":
                    if guess != c or 1 in [scores[g] for g in scores if g != c]:
                        err += 1
                else:
                    if guess != c:
                        err += 1
                self.update(f,c,guess,scores[guess],scores)
            if err == 0:
                break
            if self.verbose == True:
                print("Iteration: ", iteration, " -- TRAIN: (", err, "/", len(self.features), ") ", (len(self.features)-err)/len(self.features))
            for c,f in devexamples:
                outs = [sum([w[feat] for feat in f]) for w in self.weights]
                guess = self.classify(outs)[0]
                if guess != c:
                    deverr += 1
            if (prevdeverr < deverr or deverr == 0) and err == 0:
                break
            preverr = err
        self.trained = True
            

    # Shows the likelihood of a given example to be in each class
    def decision_function(self, features):
        if not self.trained:
            raise ValueError("Machine not trained")
        test_fs = [[self.feattoint[str(f)] for f in feats if str(f) in self.feattoint] for feats in features]
        f = test_fs
        classifications = []
        for f in test_fs:
            outs = [sum([w[feat] for feat in f]) for w in self.weights]
        return {self.inttoclass[x]:outs[x] for x in range(len(outs))}
       
    
    # Predicts the class of an example
    def predict(self, features):
        if not self.trained:
            raise ValueError("Machine not trained")
        test_fs = [[self.feattoint[str(f)] for f in feats if str(f) in self.feattoint] for feats in features]
        f = test_fs
        classifications = []
        for f in test_fs:
            outs = [sum([w[feat] for feat in f]) for w in self.weights]
            classifications.append(self.inttoclass[self.classify(outs)[0]])
        return classifications
    
if __name__ == "__main__":
    P = SGDClassifier(lossfn = "logistic", shuffle = True, averaged = True, verbose = True)

    # 4 training examples, no dev examples (can use almost any data type for feature names)
    # We simply list the 'hot' features for each example 
    features = [['w','x','y','z'], ['u','w','x'],[232,'w'],[232,'x','y','z']]

    # The corresponing classes
    classes = ['CLASS_A','CLASS_A','CLASS_B','CLASS_A']

    # Train
    P.fit(features, classes)

    # Show probabilities of the classes for an instance
    print(P.decision_function([[232,'w','z']])) # Print scores for classes

    # Show how the classes correspond to indices
    print(P.classtoint)

    # Show the best class for an example
    print(P.predict([[232,'w','z']]))