import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('large_spam1.csv', encoding='latin-1')

df = df.rename(columns={'v1': 'label', 'v2': 'text'})

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer


# 5. Vectorize Using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

y_pred = model.predict(X_test_tfidf)

print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

spam_words = ' '.join(df[df['label'] == 1]['text'])
ham_words = ' '.join(df[df['label'] == 0]['text'])

spam_wc = WordCloud(width=500, height=300).generate(spam_words)
ham_wc = WordCloud(width=500, height=300).generate(ham_words)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Spam WordCloud')

plt.subplot(1, 2, 2)
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Ham WordCloud')

plt.show()
