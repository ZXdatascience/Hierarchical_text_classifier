import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def preproc1(comment):
    """

    :param comment: String
    :return:
    """
    doc = nlp(comment)
    lemma_sentence = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(lemma_sentence).lower()

def vectorize(text):
    """

    :param text: List of String
         eg. text = ['Lebron James in the GOAT, he is the greatest basketball player in the history of basketball.',
        "President Trump's trade war against China has produced a tension between Chinese investors and US government."]
    :return:
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(t for t in text)
    return vectorizer, matrix