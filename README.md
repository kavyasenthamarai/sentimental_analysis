# sentimental_analysis_customer_review

# DEVELOPED BY

## TECH TURTLES
```
KAVYA K (AI-DS)
YUVA KRISHNA K(CS-IOT)
PRIYADHARSHINI P(CS-CYBER SECURITY)
RAMA E.K. LEKSHMI(AIML)
SWETHA P(CS-CYBER SECURITY)
```
# "Sentiment Analysis with LSTM: Predicting Product Reviews"
## Algorithm 
## Data Preprocessing:

Load the dataset containing reviews and their corresponding scores.
Select relevant columns (e.g., 'Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text').
## Sentiment Labeling:

Factorize the 'Score' column to convert it into a binary sentiment label (e.g., 'Negative' and 'Positive').
Text Tokenization and Padding:

Tokenize the text data using a tokenizer (with a specified vocabulary size, if needed).
Pad the sequences to ensure they have a fixed length for input to the model.
## Model Definition and Training:

Define a sequential model with an embedding layer, dropout layers, an LSTM layer, and a dense output layer with a sigmoid activation function.
Compile the model with binary cross-entropy loss and the Adam optimizer.
Train the model using the tokenized and padded sequences along with their corresponding sentiment labels. Monitor validation accuracy to prevent overfitting.
## Prediction and Sentiment Classification:

Create a function to predict the sentiment of a given input text.
Tokenize and pad the text, then use the trained model to predict the sentiment.
Translate the prediction into a sentiment label (e.g., 'Negative' or 'Positive') and print the result.

# Programm
# Import necessary libraries
```
pip install pandas matplotlib tensorflow
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
import matplotlib.pyplot as plt
```
![image](https://github.com/kavyasenthamarai/sentimental_analysis/assets/118668727/752d4d9e-b99a-46ce-a265-15f3022257cb)

# Data Preprocessing: Cleaning, extracting relevant columns, and encoding sentiment labels.
```
df = pd.read_excel('/content/review')
print(df.columns)
```
![image](https://github.com/kavyasenthamarai/sentimental_analysis/assets/118668727/0721b596-e422-4954-948e-887bbb2e4a63)

# Tokenization and Padding: Preprocessing text data for model input.
```
review_df = df[['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']]

print(review_df.shape)
review_df.head(5)
sentiment_label = review_df['Score'].factorize()
sentiment_label

tweet = review_df['Score'].values
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequence = pad_sequences(encoded_docs, maxlen=200)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweet)
```
# LSTM Model: Implementation and training of the LSTM neural network.
```
vocab_size = len(tokenizer.word_index) + 1

embedding_vector_length = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=10, batch_size=32)
import matplotlib.pyplot as plt
```
![image](https://github.com/kavyasenthamarai/sentimental_analysis/assets/118668727/ab178d05-f44f-4b64-b8ca-be3f05518da3)

# Training Visualization: Plots showing model performance.
```
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

plt.savefig("Accuracy plot.jpg")
```
![image](https://github.com/kavyasenthamarai/sentimental_analysis/assets/118668727/9441b557-b940-41b3-9652-09eddb8ef90d)

```
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.legend()
plt.show()

plt.savefig("Loss plt.jpg")
```
![image](https://github.com/kavyasenthamarai/sentimental_analysis/assets/118668727/2250a6c7-bf1b-4e2b-b5fa-5e0fd32cebf5)

# Prediction Function: Function to predict sentiment for user-provided text.
```
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    Score = 'Positive' if prediction == 0 else 'Negative'
    print(f"The sentiment of the sentence '{text}' is {Score}.")
```
# Example Predictions
```
test_sentence1 = "I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most"
predict_sentiment(test_sentence1)

test_sentence2 = "The candy is just red , No flavor . Just  plan and chewy .  I would never buy them again"
predict_sentiment(test_sentence2)

```
![image](https://github.com/kavyasenthamarai/sentimental_analysis/assets/118668727/2c0bfac3-3dce-450e-b943-e05c64cdd296)
