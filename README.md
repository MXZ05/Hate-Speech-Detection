# Hate Speech and Offensive Language Detection

This repository contains a machine learning project that detects hate speech, offensive language, and neutral speech in tweets. The dataset used for this project is the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

## Dataset
The dataset used for this project can be found on Kaggle: [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

The dataset consists of tweets labeled into three classes:
- 0: Hate Speech
- 1: Offensive Language
- 2: No Hate or Offensive Language

## Project Structure

- `Hate_data.csv`: This is the dataset containing tweets and their corresponding labels.
- `HateSpeechDetection.py`: The main Python script that loads, processes, and trains a machine learning model on the dataset.

## Dependencies

To run the project, make sure to install the necessary libraries:

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
```

Additionally, you'll need to download the NLTK stopwords data:

```bash
import nltk
nltk.download('stopwords')
```

## Data Processing

The project involves several key steps in data processing:

1. **Data Cleaning**: The tweets are cleaned by removing URLs, special characters, stop words, and applying stemming to reduce words to their base form.
2. **Feature Extraction**: The text is vectorized using `CountVectorizer` to convert text data into numerical features.
3. **Data Splitting**: The dataset is split into training and testing sets (80% training, 20% testing).
4. **Model Building**: A Decision Tree Classifier is used to train on the data and make predictions on test data.
5. **Evaluation**: The model’s performance is evaluated using a confusion matrix and accuracy score.

## Usage

1. Clone the repository:
    ```bash
   git clone https://github.com/MXZ05/Hate-Speech-Detection.git
    ```

2. Run the model:
    ```bash
    python HateSpeechDetection.py
    ```

3. You can input your own tweet to see if it is classified as Hate Speech, Offensive Language, or No Hate or Offensive Language.

## Example

```bash
Input tweet to be sent: "Let's unite and kill all the people who are protesting against the government"
Predicted label: ['Hate Speech']
```

## License

This project is licensed under the MIT License.

---
