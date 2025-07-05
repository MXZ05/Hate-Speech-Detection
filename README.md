# Hate Speech and Offensive Language Detection

This repository contains a machine learning project that detects **hate speech**, **offensive language**, and **neutral speech** in tweets. The dataset used for this project is the Hate Speech and Offensive Language Dataset.

---

## 📂 Dataset

The dataset used for this project can be found on Kaggle: [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/t-davidson/hate-speech-and-offensive-language).

The dataset consists of tweets labeled into three classes:

- `0`: Hate Speech  
- `1`: Offensive Language  
- `2`: No Hate or Offensive Language

---

## 🏗️ Project Structure

- `Hate_data.csv`: This is the dataset containing tweets and their corresponding labels.  
- `HateSpeechDetection.py`: The original script that loads, processes, and trains a machine learning model on the dataset.  
- `ui.py`: A Gradio-based user interface that allows users to test the hate speech detection model in real time.  
- `Info.pdf`: Background or reference document related to the dataset or project.  
- `README.md`: Project documentation and setup instructions.

---

## 📦 Dependencies

To run the project, make sure to install the necessary libraries:

```bash
pip install pandas numpy scikit-learn nltk gradio
```

Additionally, you'll need to download NLTK stopwords data:
```bash
import nltk
nltk.download('stopwords')
```

---

## 🔄 Data Processing

The project involves several key steps:

- Data Cleaning: Tweets are cleaned by removing URLs, special characters, stopwords, and applying stemming.

- Feature Extraction: Tweets are vectorized using CountVectorizer to convert text data into numerical features.

- Data Splitting: The dataset is split into 80% training and 20% testing sets.

- Model Building: A DecisionTreeClassifier is used to train on the data.

- Evaluation: The model’s performance is evaluated using metrics such as accuracy and confusion matrix.

---

## 🚀 Usage

1. Clone the repository
```bash
git clone https://github.com/MXZ05/Hate-Speech-Detection.git
cd Hate-Speech-Detection
```

2. Run the model from terminal
```bash
python HateSpeechDetection.py
```

3. Or run the interactive Gradio UI
```bash
python ui.py
```
This will open a browser window where you can enter any tweet and instantly get a prediction.

---

## 🧪 Example

Input tweet:
```bash
"Let's unite and kill all the people who are protesting against the government"
```

Predicted label:
```bash
['Hate Speech']
```

---

## 🌐 Gradio UI Preview

The Gradio interface provides a clean and modern layout where users can:

- Enter any tweet or comment
- Instantly get predictions in one of the following categories:

    - 🛑 Hate Speech
    - ⚠️ Offensive Language
    - ✅ No Hate or Offensive Language

---

## 📄 License

This project is licensed under the MIT License.

---
