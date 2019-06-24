from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import wordnet
from nltk import NaiveBayesClassifier
import pickle
import random

classifier = []
opinionList = []

try : 
    classifierFile = open("file.pickle", "rb")
    classifier = pickle.load(classifierFile)
    classifierFile.close()
except :
    positive = sent_tokenize(open("positive.txt", "r").read())
    negative = sent_tokenize(open("negative.txt", "r").read())

    all_words = []
    documents = []

    for positive_sentence in positive :
        all_words = all_words+word_tokenize(positive_sentence) #list + list mknya ga append
        documents.append((word_tokenize(positive_sentence),"pos"))
        
    for negative_sentence in negative :
        all_words = all_words+word_tokenize(negative_sentence) 
        documents.append((word_tokenize(negative_sentence),"neg"))

    all_words = [word.lower() for word in all_words] #for word in all_words, all_words.append(word.lower())
    random.shuffle(documents)
    #[:1000] ambil s/d index ke 1000
    word_features = list(set(all_words))[:1000] #set itu buat ngilangin duplikat, abis itu di list lg buat balik ke list
    
    def find_features(wordList) :
        words = set(wordList)
        features = {}
        for w in word_features :
            features[w] = (w in words)
        return features

    training_set = []
    
    for wordList, category in documents :
        training_set.append((find_features(wordList),category))
    
    classifier = NaiveBayesClassifier.train(training_set)

while True:
    choose = 0
    print("Opinion List")
    print("============")
    
    if len(opinionList) > 0 :
        for i, opinion in enumerate(opinionList) :
            print(str(i+1), ". ", opinion)
    else :
        print ("No opinion inserted")

    print("")
    print("Opinion Analysis")
    print("1. Insert Opinion")
    print("2. Analyze Opinion")
    print("3. Exit")
    
    try :
        choose = int(input("Choose[1-3] : ")) #if input is not integer, it will break, which will execute the except :
    except :
        print("Input must be numeric")
        choose = 0

    if choose == 1 :
        while True :
            opinion = input("Input your opinion [5-30] : ")
            if len(opinion) > 5 and len(opinion) < 30 :
                opinionList.append(opinion)
                print("Opinion inserted successfully")
                break
            else :
                print("input must be between 5 and 30")
    elif choose == 2:
        opinionIndex = -1
        while True :
            opinionIndex = int(input("Choose Opinion [1-"+str(len(opinionList))+"] : "))
            if opinionIndex >= 1 and opinionIndex <= len(opinionList) :
                break
            else :
                print("Input must be between 1 and "+str(len(opinionList)))
        opinion = opinionList[opinionIndex-1]

        p = 0
        n = 0

        for word in word_tokenize(opinion) :
            category = classifier.classify(FreqDist(word))

            if category == "pos" :
                p+=1
            else :
                n+=1

        print("Your opinion is categorized as ")
        if p>n :
            print("positive")
        elif p<n :
            print("negative")
        else :
            print("neutral")

        showAnalysis = ""
        while True :
            showAnalysis = input("Show Analysis result [yes/no] <case sensitive> : ")
            if showAnalysis == "yes" or showAnalysis == "no" :
                break
        
        if showAnalysis == "yes" :
            wordList  = word_tokenize(opinion)
            freqDistList = FreqDist(wordList).most_common()
            posTagList = pos_tag(wordList)
            chunkTree  = ne_chunk(posTagList)

            for word in wordList :
                print("Word : " + word)
                #posTag contains : ['word', tag]. ex : ['eat', VB]. Thats why we need to loop it
                for posTag in posTagList :
                    if posTag[0] ==  word :
                        print("Tag : " + posTag[1])

                print("antonym :")

                for synset in wordnet.synsets(word) :
                    for lemma in synset.lemmas() :
                        for antonym in lemma.antonyms() :
                            print (" = " + antonym.name())

                for freqDist in freqDistList :
                    if freqDist[0] == word :
                        print("Frequency : " + str(freqDist[1]))

        print("Press Enter to Continue...")
        input()
        chunkTree.draw()
    elif choose == 3 :
        classifierFile = open("file.pickle","wb")
        pickle.dump(classifier,classifierFile)
        classifierFile.close()
        break
    else :
        print("Input must be between 1 and 3")

    
    


