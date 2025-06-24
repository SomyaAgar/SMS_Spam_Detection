# Spam / Not Spam Classifier
## Project Overview
This project is a machine learning classifier built to distinguish between spam and not spam (ham) messages. The aim is to develop an efficient and accurate model that can automate email or SMS filtering using natural language processing (NLP) techniques.

## Technologies Used
- Python 3
- Pandas – for data manipulation
- NumPy – for numerical operations
- Scikit-learn – for model training and evaluation
- NLTK – for text preprocessing
- Matplotlib And Seaborn – for visualization
- Streamlit- For deployment
- Jupyter Notebook – for experimentation and documentation

## Models Used
- LogisticRegression
- Support Vector Classifier
- Multinomial Naive Bias
- DecisionTree Classifier
- KNeighbors Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Bagging Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- XGB Classifier
- Model comparison was based on accuracy and precision.

## Dataset
The dataset used is the SMS Spam Collection Dataset from Kaggle: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset]
- Features: SMS message text
- Target: spam or ham

## Data Preprocessing
- Lowercasing
- Removing punctuation and stopwords
- Tokenization
- Stemming
- Vectorization using TF-IDF

## Results
| Algorithm | Accuracy  | Precision |
|-----------|-----------|-----------|
| KNN       | 0.896518  | 1.000000  |
| NB        | 0.957447  | 1.000000  |
| ETC       | 0.974855  | 0.982759  |
| RF        | 0.971954  | 0.965812  |
| SVC       | 0.966151  | 0.963964  |
| GBDT      | 0.943907  | 0.944444  |
| LR        | 0.951644  | 0.940000  |
| XGB       | 0.964217  | 0.931624  |
| AdaBoost  | 0.896518  | 0.897436  |
| BgC       | 0.961315  | 0.895161  |
| DT        | 0.928433  | 0.872093  |
-------------------------------------

## Future Improvements
- Integrate with email APIs for real-time classification
- Use deep learning models (e.g., LSTM or BERT)
