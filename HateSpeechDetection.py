# Importing libraries
import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv("Hate_data.csv")

#print(dataset.isnull().sum())
#print(dataset.info()) 
#print(dataset.describe())

# Add a 'labels' column that maps numerical class to meaningful text labels
dataset["labels"] = dataset["class"].map({0: "Hate Speech",
                                          1: "Offensive Language",
                                          2: "No Hate or Offensive Language"})

#print(dataset)

# Select only the 'tweet' and 'labels' column
data = dataset[["tweet","labels"]]

#print(data)

import re
import nltk
import string

#Importing of stop words
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

#Importing of stemming
stemmer = nltk.SnowballStemmer("english")


# Define a function to clean the text data
def clean_data(text):
    text = str(text).lower()
    text = re.sub(r"http?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)  
    text = re.sub(r"rt", "", text)  
    
    #Stopwords removal
    text = [word for word in text.split(" ") if word not in stopwords]
    text = " ".join(text)

    #Stemming the text
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)

    return text

# Apply the 'clean_data' function to clean the 'tweet' column
data.loc[:, "tweet"] = data["tweet"].apply(clean_data)

#print(data)

# Convert the 'tweet' and 'labels' columns to numpy arrays
x = np.array(data["tweet"])
y = np.array(data["labels"])
#print(x)
#print(y)

# Import libraries for vectorizing text and splitting data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
x = cv.fit_transform(x)
#print(x)

# Split data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print(x_train)

#Building ML model
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

#Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#print(cm)

# Visualize the confusion matrix using a heatmap
import seaborn as sms
import matplotlib.pyplot as ply


sms.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
#ply.show()

from sklearn.metrics import accuracy_score

# Calculate the accuracy of the model
accuracy_score(y_test,y_pred)
#print(accuracy_score(y_test,y_pred))

#sample = "Let's unite and kill all the people who are protesting against the government"
#sample = clean_data(sample)
#print(sample)

#data1 = cv.transform([sample]).toarray()
#print(data1)

#dt.predict(data1)
#print(dt.predict(data1))

# Sample testing using user input
test_tweet = str(input("Input tweet to be sent : "))
test_tweet = clean_data(test_tweet)
test_data = cv.transform([test_tweet]).toarray()
dt.predict(test_data)
print(dt.predict(test_data))