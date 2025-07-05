# Importing libraries
import pandas as pd
import numpy as np
import gradio as gr
import re
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
dataset = pd.read_csv("Hate_data.csv")

# Add a 'labels' column mapping numerical classes to meaningful text labels
dataset["labels"] = dataset["class"].map({0: "🛑 Hate Speech",
                                          1: "⚠️ Offensive Language",
                                          2: "✅ No Hate or Offensive Language"})

# Select only the 'tweet' and 'labels' column
data = dataset[["tweet", "labels"]]

# Initialize NLTK components
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

# Define a function to clean the text data
def clean_data(text):
    text = str(text).lower()
    text = re.sub(r"http?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>+", "", text)  # Remove HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  # Remove punctuation
    text = re.sub(r"\n", "", text)  # Remove newlines
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words containing numbers
    text = re.sub(r"rt", "", text)  # Remove 'rt' (retweet)
    
    # Remove stopwords
    text = " ".join([word for word in text.split(" ") if word not in stopwords])
    
    # Stemming the text
    text = " ".join([stemmer.stem(word) for word in text.split(" ")])
    
    return text

# Apply the 'clean_data' function to clean the 'tweet' column
data["tweet"] = data["tweet"].apply(clean_data)

# Convert the 'tweet' and 'labels' columns to numpy arrays
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Vectorize text
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Function to predict hate speech
def detect_hate_speech(text):
    text = clean_data(text)  # Preprocess input text
    vectorized_text = cv.transform([text])  # Transform text using CountVectorizer
    prediction = dt.predict(vectorized_text)  # Make prediction
    return prediction[0]  # Return the predicted category

# Create the Gradio interface with a modern look
iface = gr.Interface(
    fn=detect_hate_speech,
    inputs=gr.Textbox(lines=3, placeholder="Enter a tweet...", label="Input Text"),
    outputs=gr.Label(label="Prediction"),
    title="Hate Speech & Offensive Language Detector",
    description="""
### 🔍 Detect Hate Speech & Offensive Language  
This AI model classifies text into three categories:  
- 🛑 **Hate Speech**  
- ⚠️ **Offensive Language**  
- ✅ **No Hate or Offensive Language**  

 **Try it out!** Type or paste any text and get instant classification.
""",
    theme="compact",
    examples=[
        ["I hate people of a certain race."],
        ["You are so stupid!"],
        ["Hope you have a great day!"],
        ["Get out of my country!"],
        ["I support equal rights for all."]
    ]
)

# Launch the interface
iface.launch()
