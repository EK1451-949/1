from gensim import corpora
from gensim import models
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "Ice cream is a yummy treat durning summer break",
    "Ice comes in many flavors like chocolate, strawberry, and more",
    "Tennis is an interesting spot that requires lots of energy",
    "Tennis players increase energy by training everyday",
    "However, in the summer it is to hot to play tennis"
]

def preprocess(documents):
    #tokanize the text into words
    docs = []
    for doc in documents:
        words = word_tokenize(doc)
        stop_words = ["a", "of", "in", "to", "and", "for", "the"]
        punc =[".", ",","'", "(", ")"]

        good_words = []
        for word in words:  
            if  word.casefold() not in stop_words and ord not in punc: 
                good_words.append(word)
        docs.append(good_words)
    
    freq = defaultdict(int)
    for doc in docs:
        for word in doc:
            freq[word] += 1
    docs = [ [ word for word in text if freq[word] > 1 ] for text in docs]
    return docs

texts = preprocess(documents)
dictionary = corpora.Dictionary(texts)
corpus = [ dictionary.doc2bow(text) for text in texts]

# Latent semantic indexing
def LSI(corpus, dictionary):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    corpus_lsi = lsi_model[corpus_tfidf]

    for number, (topics, original_text) in enumerate(zip(corpus_lsi, documents), 1):

        print(f"\nDocument {number}")
        print("-" * 70)
        print(f"Text: {original_text}")
        print("Topic scores:")
        for topic_number, score in topics:
            print(f"  Topic {topic_number + 1}: {score:.4f}")

    print("\n" + "=" * 70)
    print("TOPIC KEYWORDS")
    print("=" * 70)

    for topic_number, topic_words in lsi_model.print_topics():
        print(f"\nTopic {topic_number + 1}:")
        print(topic_words)

# LSI(corpus, dictionary)

#Latent Dirichlet Allocation
def LDA(corpus, dictionary):
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=3)
    corpus_lda = lda_model[corpus]

    for number, (topics, original_text) in enumerate(zip(corpus_lda, documents), 1):
        print(f"\nDocument {number}")
        print("-" * 70)
        print(f"Text: {original_text}")
        print("Topic scores:")
        for topic_number, score in topics:
            print(f"  Topic {topic_number + 1}: {score:.4f}")

    print("\n" + "=" * 70)
    print("TOPIC KEYWORDS")
    print("=" * 70)

    for topic_number, topic_words in lda_model.print_topics():
        print(f"\nTopic {topic_number + 1}:")
        print(topic_words)

LDA(corpus, dictionary)
