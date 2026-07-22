import nltk

# nltk.download("punkt_tab")
# nltk.download("stopwords")

import string

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from gensim import corpora, models

reviews = [
    "This game had insanely good graphics and the animation was great.",
    "The storyline of the game was too slow.",
    "The storyline was slow but well worth it since it had amazing charater development.",
    "Best game I've played since it had good action, charaters, and gameplay.",
    "This game has terrible loading times and took forever to get into it.",
    "Overall game is fun but there are some bugs that ruin the experience since it softlocks the player.",
    "I love how all the charaters get a full backstory and are fully fleshed out instead of the game only focusing on the main charater.",
    "Seeing how this game was made by only one person with limited funds, it deserves to be recognized.",
    "This is a terrible game for children since it contains lots of violence and gore."
]

def preprocess(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))

    clean_words = []

    for word in words:
        if (
            word not in stop_words
            and word not in string.punctuation # import string
            and word.isalpha()
        ):
            clean_words.append(word)

    return clean_words


def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity


def get_rating(polarity):
    if polarity <= -0.8:
        return 1
    elif polarity <= -0.3:
        return 2
    elif polarity < 0.0:
        return 3
    elif polarity < 0.5:
        return 4
    else:
        return 5

def show_stars(rating):
    return "★" * rating + "☆" * (5 - rating)

def get_opinion(polarity):
    if polarity >= 0.5:
        return "Very Positive"
    elif polarity > 0.1:
        return "Positive"
    elif polarity >= -0.1:
        return "Neutral"
    elif polarity > -0.5:
        return "Negative"
    else:
        return "Very Negative"

def create_lda(reviews):
    cleaned_reviews = []
    for review in reviews:
        cleaned_reviews.append(preprocess(review))

    dictionary = corpora.Dictionary(cleaned_reviews)

    corpus = []
    for review in cleaned_reviews:
        corpus.append(dictionary.doc2bow(review))

    lda_model = models.LdaModel(
        corpus, id2word=dictionary,
        num_topics=3, passes=20,
        random_state=42, minimum_probability=0
    )

    return lda_model, corpus

def analyze_reviews(reviews, lda_model, corpus):
    rating_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }

    total_polarity = 0

    print("\n" + "=" * 70)
    print("REVIEW ANALYSIS")
    print("=" * 70)

    for i in range(len(reviews)):
        sentiment, polarity = get_sentiment(reviews[i])
        rating = get_rating(polarity)

        # Find the strongest LDA topic
        topic_scores = lda_model[corpus[i]]
        main_topic = max(topic_scores, key=lambda item: item[1])

        topic_number = main_topic[0]
        topic_score = main_topic[1]

        topic_words = lda_model.show_topic(topic_number, topn=5)
        keywords = []
        for word, score in topic_words:
            keywords.append(word)

        rating_counts[rating] += 1
        total_polarity += polarity

        print(f"\nReview {i + 1}")
        print("-" * 70)
        print("Text:", reviews[i])
        print("Sentiment:", sentiment)
        print(f"Polarity Score: {polarity:.2f}")
        print("Rating:", show_stars(rating))
        print("Main Topic:", topic_number + 1)
        print("Topic Keywords:", ", ".join(keywords))
        print(f"Topic Score: {topic_score:.2f}")

    average_polarity = total_polarity / len(reviews)
    average_rating = get_rating(average_polarity)

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print("Total Reviews:", len(reviews))
    print(f"Average Polarity: {average_polarity:.2f}")
    print("Overall Opinion:", get_opinion(average_polarity))
    print("Overall Rating:", show_stars(average_rating))

    return rating_counts

def analyze_top_words(reviews):
    word_reviews = defaultdict(list)

    for review_number in range(len(reviews)):
        words = preprocess(reviews[review_number])

        # Count each word only once per review
        unique_words = set(words)

        for word in unique_words:
            word_reviews[word].append(review_number)

    results = []

    for word, review_numbers in word_reviews.items():
        if len(review_numbers) >= 2:
            total_polarity = 0

            for number in review_numbers:
                polarity = TextBlob(
                    reviews[number]
                ).sentiment.polarity

                total_polarity += polarity

            average = total_polarity / len(review_numbers)

            results.append(
                (word, len(review_numbers), average)
            )

    # Sort by how many reviews contain the word
    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # Keep only the top five words
    results = results[:5]

    print("\n" + "=" * 70)
    print("TOP 5 COMMON WORD TOPICS")
    print("=" * 70)

    for word, count, average in results:
        print(f"\nTopic: {word}")
        print("-" * 70)
        print("Appears In:", count, "reviews")
        print(f"Average Polarity: {average:.2f}")
        print("Overall Opinion:", get_opinion(average))

def main():
    # Create the LDA model
    lda_model, corpus = create_lda(reviews)

    # Show only the five most common words
    analyze_top_words(reviews)

    # Analyze each review
    rating_counts = analyze_reviews(
        reviews,
        lda_model,
        corpus
    )


main()