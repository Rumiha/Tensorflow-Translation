import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tensorflow import keras

SHOW_TRAINING_GRAPHS = False

def main():
    #Load raw data
    rawData = pd.read_csv("data.csv")

    #Separating X and Y data
    x, y_string = getXYdata(rawData)
    classesArray = []
    for sss in y_string.values:
        if sss not in classesArray:
            classesArray.append(sss)
    y = []
    for i in range(len(y_string.values)):
        y.append(classesArray.index(y_string.values[i]))

    #Making train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)

    #Model
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x.values)
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(67)
    ])
    model.compile(optimizer='adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    

    #Train model
    modelX = model.fit(np.array(x.values), np.array(y), epochs=100)
    
    #Graphs
    if SHOW_TRAINING_GRAPHS==True:
        plt.plot(modelX.history['accuracy'])
        plt.plot(modelX.history['val_accuracy'])
        plt.show()


    #Testing
    TEST_RIJEC = "volibol"

    aaa = [TEST_RIJEC, "hmm", "hmm"]
    bbb = np.zeros((3,20))
    for i in range(3):
        bbb[i] = normalize(aaa[i])
    accuracy = model.predict(bbb[0])
    print('Test accuracy :', accuracy[0])
    result = np.where(accuracy[0] == np.amax(accuracy[0]))
    print(classesArray[result[0][0]] + ", id: " + str(result[0][0]) + ", accuracy: "+ str(accuracy[0][result[0][0]]))



def normalize(word):
    normalizedWord = np.zeros(20)
    #Number of letters
    normalizedWord[0] = len(word)

    #First letter
    normalizedWord[1] = ord(word[0].lower())

    #Last letter
    normalizedWord[2] = ord(word[-1].lower())

    #Second letter
    normalizedWord[3] = ord(word[1].lower())

    #Second-last letter
    normalizedWord[4] = ord(word[-2].lower())    


    #Number of vowels
    vowel = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    count = 0
    for letter in word:
        if letter in vowel:
            count += 1
    normalizedWord[5] = float(count)

    #Number of letters E, T, O
    for letter in word:
        if letter == 'e' or letter == 'E':
            normalizedWord[6] += 1

        if letter == 't' or letter == 'T':
            normalizedWord[7] += 1

        if letter == 'o' or letter == 'O':
            normalizedWord[8] += 1

    #Same letters, multiple letters
    chars = "abcdefghijklmnopqrstuvwxyz"
    for k in range(26):
        count = word.count(chars[k])
        if count > 1:
            normalizedWord[9] += 1

    #Following latters
    for i in range(len(word)-1):
        #Latter D followed by vowel
        if word[i].lower() == 'd' and word[i+1] in vowel:
            normalizedWord[10] += 1

        #Latter B followed by vowel    
        if word[i].lower() == 'b' and word[i+1] in vowel:
            normalizedWord[11] += 1

        #Latter C followed by vowel
        if word[i].lower() == 'c' and word[i+1] in vowel:
            normalizedWord[12] += 1

        #Latter H followed by vowel
        if word[i].lower() == 'h' and word[i+1] in vowel:
            normalizedWord[13] += 1

        #Latter J followed by vowel
        if word[i].lower() == 'j' and word[i+1] in vowel:
            normalizedWord[14] += 1
    if len(word)>5:
        for i in range(5):
            #Few Letters
            normalizedWord[i+15] = ord(word[i].lower())
    else:
        for i in range(len(word)):
            normalizedWord[i+15] = ord(word[i].lower())

    return normalizedWord

def getXYdata(rawData):
    rawx = rawData.drop('cro', axis=1)
    rawy = rawData.drop('foreign', axis=1)

    #Make array to store normalized data
    dataRaw = pd.read_csv("dataold.csv")
    data = dataRaw.to_numpy()
    indexCounter = 0
    indexChecker = True
    correctedData = np.zeros((1742, 2), dtype=object)
    for line in data:
        indexChecker = True
        croWord = ""
        for word in line:
            if indexChecker==True:
                croWord = word
                indexChecker = False
            correctedData[indexCounter][0] = croWord
            correctedData[indexCounter][1] = word
            indexCounter += 1

    npdatay = np.array(correctedData[:, 0])
    pddatay = pd.DataFrame(npdatay, columns=["cro"]) 

    #Making 2 column data
    trainData = np.zeros((1742, 2, 20))
    for i in range(1742):
        for j in range(2):
            if correctedData[i][j] != 0:
                trainData[i][j] = normalize(correctedData[i][j])

    #Separating train data
    dataLabels = trainData[:, 0]
    dataFeatures = trainData[:, 1]


    npdatax = np.array(dataFeatures)

    pddatax = pd.DataFrame(npdatax, columns=["a", "b", "c","d","e","f","g","h","i","j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "z"])

    return pddatax, pddatay


if __name__=="__main__":
    main()
