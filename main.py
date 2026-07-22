import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

def get_sentiment(text):
    # Use TextBlob for sentiment analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def sentiment_analysis(text):
    print("Target String:", text)
    print("Sentiment:", get_sentiment(text))

# txt = "Mister, usually written in its contracted form Mr., is a commonly used English honorific for men without a higher honorific, or professional title, or any of various designations of office. The title 'Mr.' derived from earlier forms of master, as the equivalent female titles Mrs., Miss, and Ms. all derived from earlier forms of mistress. Master is sometimes still used as an honorific for boys and young men."
txt = "I'm mad because my ice cream melted. I worked hard to get enough money for my ice cream but now it is gone forever. However, my dad told me it was okay and he bought me another cone."


sent =sent_tokenize(txt)

for s in sent:
    sentiment_analysis(s)
    print("_____________________________")


# stop_words = set(stopwords.words("english"))
# punc =[".", ",","'"]

# good_words = []

# for word in words:  
#     if word.casefold() not in stop_words and word not in punc:
#         good_words.append(word)
#         # print(word)

# reduced =[]
# for word in good_words:
#     if word.casefold() not in reduced:
#         reduced.append(word)

# print(reduced)





# for item in txt.split("."):
#     print(item)