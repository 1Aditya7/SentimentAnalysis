# **Amazon Alexa Review Sentiment Analysis**

## **Introduction**

The project aims to analyze user reviews of Amazon Alexa products, classify them as positive or negative based on user feedback, and predict the sentiment of reviews using machine learning models. The dataset used consists of user ratings, feedback labels, product variations, and textual reviews.

The dataset used in this project is available at: [Amazon Alexa Reviews Dataset](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews).

The objective is to preprocess the text data, build a model pipeline, and evaluate the performance of different classifiers (Random Forest, XGBoost, and Decision Tree) for sentiment prediction.

## **Methodology**

### **Data Preprocessing**
- The dataset was loaded and examined for missing values.
- A new column, `length`, was created to represent the length of each review.
- The dataset was analyzed to understand the distribution of ratings, feedback, and product variations.

### **Text Preprocessing**
- Text reviews were cleaned by removing non-alphabetic characters, converting text to lowercase, and splitting them into words.
- The stopwords were removed, and the remaining words were stemmed using the Porter Stemmer.
- The corpus was built for use in the model.

### **Feature Engineering**
- A `CountVectorizer` was applied to convert the text data into a bag-of-words representation.
- A total of 2500 features were selected using the `CountVectorizer`.

### **Data Splitting and Scaling**
- The dataset was split into a training and testing set (80-20 split).
- Features were scaled using `MinMaxScaler` to ensure that the values were between 0 and 1.

### **Modeling**
Multiple classifiers were trained and evaluated, including:
1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Decision Tree Classifier**

Each model's performance was evaluated based on training and testing accuracy, confusion matrix, and cross-validation scores.

## **EDA (Exploratory Data Analysis)**

- **Rating Distribution**: The majority of reviews have a rating of 5 stars (72.57%), followed by 4 stars (14.44%) and lower ratings.
  
  ![Rating Distribution](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/reviewDistribution.png)<br>
  *Inference*: Most users rate the product highly (5 stars), suggesting general satisfaction with the product. This high concentration of 5-star reviews may lead to an imbalanced dataset that could affect model performance, potentially biasing it toward predicting positive sentiments.

- **Feedback Distribution**: A significant portion (91.84%) of feedback labels is positive (feedback = 1), with a smaller portion being negative (feedback = 0).
  
  ![Feedback Distribution](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/feedbackDistribution.png)<br>
  *Inference*: The feedback data is imbalanced, with most reviews being positive. This imbalance is important to consider when building models, as it might result in the model being more likely to predict positive feedback due to the higher prevalence of positive reviews in the dataset.

- **Variation Distribution**: The product variation with the highest count is 'Black Dot' (16.38% of the dataset), followed by 'Charcoal Fabric' (13.65%).
  
  ![Variation Distribution](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/variationDistribution.png)<br>
  *Inference*: Variations like 'Black Dot' and 'Charcoal Fabric' are more popular, indicating the preference for specific product models. This insight could be leveraged for targeted marketing or sales strategies, as customers may have a higher preference for certain variations.

- **Review Length**: Reviews generally have an average length of around 132 characters, with some extremely long reviews reaching up to 2853 characters.
  
  ![Review Length Distribution](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/reviewlenDistribution.png)<br>
  *Inference*: There is a diverse range of review lengths, with the median review length around 74 characters. Shorter reviews may reflect users who are either more concise or may have less detailed feedback, while longer reviews could be indicative of more in-depth satisfaction or dissatisfaction. The correlation between review length and sentiment could provide further insights for improving sentiment analysis models.

## **Approach**

1. **Text Processing**: The reviews were processed and transformed using the `CountVectorizer` to convert the text into a format suitable for machine learning models. Stopwords were excluded and words were stemmed to improve the model's ability to focus on key terms.
   
2. **Modeling**: 
   - Three different models were trained:
     1. **Random Forest Classifier**
     2. **XGBoost Classifier**
     3. **Decision Tree Classifier**
   - Hyperparameter tuning was performed using Grid Search to optimize the Random Forest model.
   - Accuracy, cross-validation scores, and confusion matrices were used to evaluate model performance.

3. **Evaluation**: Performance metrics, including training accuracy, testing accuracy, and confusion matrices, were used to assess each model's effectiveness in predicting the sentiment (positive or negative) of reviews.

### **Results**

#### **Random Forest Classifier**
- **Training Accuracy**: 99.37%
- **Testing Accuracy**: 93.97%
- **Cross-validation mean accuracy**: 93.85%
- **Best Hyperparameters**: 
  - `bootstrap`: True
  - `max_depth`: 100
  - `min_samples_split`: 8
  - `n_estimators`: 300
- **Confusion Matrix**: 
  ![Random Forest Confusion Matrix](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/rfcConfusionMatrix.png)<br>
  *>Inference*: The Random Forest model shows high performance on both training and testing datasets. The confusion matrix reveals a relatively low number of misclassifications, indicating the model's robustness. However, further tuning might help improve recall for the minority class.

#### **XGBoost Classifier**
- **Training Accuracy**: 97.42%
- **Testing Accuracy**: 93.33%
- **Confusion Matrix**:
  ![XGBoost Confusion Matrix](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/xgbConfusionMatrix.png)<br>
  *Inference*: XGBoost performs well, slightly outperforming the Random Forest model on training accuracy, but similar performance on testing data. The confusion matrix highlights that the model is effective at identifying both positive and negative reviews with minimal errors.

#### **Decision Tree Classifier**
- **Training Accuracy**: 99.37%
- **Testing Accuracy**: 92.22%
- **Confusion Matrix**:
  ![Decision Tree Confusion Matrix](https://github.com/1Aditya7/SentimentAnalysis/blob/main/amazonPlots/dtcConfusionMatrix.png)<br>
  *Inference*: The Decision Tree model is also performing well but has a slightly lower testing accuracy than Random Forest and XGBoost. The confusion matrix shows some misclassifications, particularly in distinguishing between positive and negative reviews.

## **Error Evaluation Metrics**

The confusion matrices for each model highlight how well the models distinguish between positive and negative reviews:

- **False Positives (FP)**: Reviews predicted as positive but are actually negative.
- **False Negatives (FN)**: Reviews predicted as negative but are actually positive.
- **True Positives (TP)**: Reviews correctly predicted as positive.
- **True Negatives (TN)**: Reviews correctly predicted as negative.

## **Accuracy Evaluation Metrics**

The evaluation metrics include:
- **Accuracy**: The overall percentage of correct predictions made by the model.
- **Precision, Recall, and F1-Score**: These metrics can be calculated using the confusion matrix and provide more insights into model performance.

## **Limitations and Future Scope**

- **Data Imbalance**: The dataset is imbalanced with a higher percentage of positive reviews. Future work could include methods like oversampling, undersampling, or using weighted loss functions to improve classification for the minority class.
- **Model Improvement**: The current models can be further fine-tuned using advanced techniques such as ensemble learning or deep learning models (e.g., LSTM or BERT for sentiment analysis).
- **Review Complexity**: The project could be expanded to analyze sentiment based on product-specific features and review sentiment trends over time.

## **Conclusions**

The project demonstrates the ability to classify Amazon Alexa product reviews into positive and negative categories effectively using machine learning models. The Random Forest, XGBoost, and Decision Tree models all perform well, with Random Forest yielding the highest accuracy on the test set. This analysis can help businesses assess customer sentiment and make informed decisions on product development.
