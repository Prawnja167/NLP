from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.corpus import wordnet, stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, accuracy
import random
import pickle
import string

def cls() :
    for i in range(10) :
        print("")

classifier = []
opinionList = []

try :
    classifierFile = open("file.pickle", "rb")
    classifier = pickle.load(classifierFile)
    classifierFile.close()
except :
    pos_sent = sent_tokenize(open("positive.txt","r").read())
    neg_sent = sent_tokenize(open("negative.txt", "r").read())

    all_words = []
    document = []
    for positive in pos_sent :
        all_words = all_words+word_tokenize(positive)
        document.append((word_tokenize(positive),"pos"))
    for negative in neg_sent :
        all_words = all_words+word_tokenize(negative)
        document.append((word_tokenize(negative),"neg"))

    all_words = [word.lower() for word in all_words]
    random.shuffle(document)
    word_feature = list(set(all_words))[:1000]


    def extract_features(wordList) :
        words = set(wordList)
        features = {}
        for w in wordList :
            features[w] = (w in words)
        return features

    training_set = []
    for wordList, categories in document :
        training_set.append((extract_features(wordList),categories))

    classifier = NaiveBayesClassifier.train(training_set)

while True :
    choose = 0
    cls()
    if len(opinionList) > 0 :
        for i,opinion in enumerate(opinionList) :
            print(i+1,". ",opinion)
    else :
        print("no opinion")
    
    print("=========")
    print ("1. input")
    print("2. analyze")
    print("3. exit")

    try :
        choose = int(input("choose menu : "))
    except :
        print("input must be numeric")
        choose = 0

    if choose == 1 :
        while True :
            opinion = input("insert opinion [5-30] :")
            if len(opinion) >= 5 and len(opinion) <= 30 :
                opinionList.append(opinion)
                break
            else :
                print("not long enough")

    elif choose == 2 :
        if len(opinionList) > 0 :
            for i, opinion in enumerate(opinionList) :
                print(i+1,". ",opinion)
        else :
            print("insert opinion first")
        
        index = -1
        while True :
            index = int(input("choose opinion [1-"+str(len(opinionList))+"] :"))
            if index > 0 and index <= len(opinionList) :
                break
            else :
                print("must choose opinion [1-"+str(len(opinionList))+"]")

        op = opinionList[index-1]
        string_no_punct = ""
        for c in op :
            if c not in string.punctuation :
                string_no_punct += c

        string_no_stopwords = []
        for word in word_tokenize(string_no_punct) :
            if word not in set(stopwords.words("english")) :
                string_no_stopwords.append(word)
        
        string_after_stem = []
        porter = PorterStemmer()
        for word in string_no_stopwords :
            temp_word = porter.stem(word)
            string_after_stem.append(temp_word)
    
        string_after_lemmatize = []
        lemma = WordNetLemmatizer()
        for word in string_after_stem :
            temp_word = lemma.lemmatize(word)
            string_after_lemmatize.append(temp_word)
        
        opinion = ""
        for word in string_after_lemmatize :
            opinion = opinion + " " + word
        
        p = 0
        n = 0

        for word in word_tokenize(opinion) :
            category = classifier.classify(FreqDist(word))
            if category == "pos" :
                p+=1
            else :
                n+=1

        print("opinion result is ", end="")
        if p>n :
            print("positive")
        elif p<n :
            print("negative")
        else :
            print("neutral")

        analysis = ""
        while True :
            analysis = input("show analysis? yes|no : ")
            if analysis == "yes" or analysis == "no" :
                break
            else : 
                print("invalid input")
        
        if analysis == "yes" :
            wordList = word_tokenize(opinion)
            posTagList = pos_tag(wordList)
            freqDistList = FreqDist(wordList).most_common()
            chunkTree = ne_chunk(posTagList)

            for word in wordList :
                print("word = ", word)
                for posTag in posTagList :
                    if posTag[0] == word :
                        print("tag = ", posTag[1])
                for freqDist in freqDistList :
                    if freqDist[0] == word :
                        print("freq = ", str(freqDist[1]))
                for synset in wordnet.synsets(word) :
                    for lemma in synset.lemmas() :
                        for antonym in lemma.antonyms() :
                            print("antonym = ", antonym.name())
            chunkTree.draw()

    elif choose == 3 :
        classifierFile = open("file.pickle", "wb")
        pickle.dump(classifier, classifierFile)
        classifierFile.close()
        break
    else :
        print("input between 1-3")