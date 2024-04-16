import nltk
nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords
from nltk import re
import pandas as pd

spacy.load('fr')
from spacy.lang.fr import French

parser = French()

fr_stop = set(nltk.corpus.stopwords.words('french'))
# v1 : numbers
my_fr_stop = fr_stop.union({'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix',
                            'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
                            'vingt', 'trente', 'quarante', 'cinquante', 'soixante', 'cent'}, fr_stop)
# v2 : conj + det + verbs
my_fr_stop = fr_stop.union({'ce', 'celui', 'cette', 'cet', 'celui-là', 'celui-ci',
                            'le', 'la', 'les', 'de', 'des', 'du',
                            'mais', 'où', 'et', 'donc', 'or', 'ni', 'car', 'depuis', 'quand', 'que', 'qui', 'quoi',
                            'ainsi', 'alors', 'avant', 'après', 'comme',
                            'être', 'avoir', 'faire',
                            'autre'})

nlp = spacy.load("fr_core_news_sm")



class ArticlePreprocessor:
    def __init__(self, article_df):
        self.article_df = article_df
        self.title_tokens = []
        self.text_tokens = []

    def fully_preprocess(self):
        for t in self.article_df.title:
            tokens = self.prepare_text(t)
            self.title_tokens.append(tokens)
        for t in self.article_df.text:
            tokens = self.prepare_text(t)
            self.text_tokens.append(tokens)

    def prepare_text(self, text):
        """
        Input:
        ------
        text: string, raw text

        Output:
        ------
        tokens: list of string, tokenized, filtered and lemmatized words from the input text
        """
        tokens = self.tokenize(text)  # split and lower case
        tokens = [re.sub(r'\b\d+\b', '', token) for token in tokens]  # get rid of digits
        tokens = [token for token in tokens if len(token) > 4]  # arbitrary length, +get rid of empty strings
        tokens = [token for token in tokens if token not in my_fr_stop]  # stopwords
        doc = nlp(' '.join(tokens))  # pave the wave for spacy lemmatizer
        tokens = [token.lemma_ for token in doc]  # obtain lemmas
        return tokens

    def tokenize(self, text):
        lda_tokens = []
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif '@' in str(token):
                lda_tokens += str(token).split('@')
            else:
                lda_tokens.append(token.lower_)
        return [t for t in lda_tokens if len(str(t)) > 0]




def main():
    news_df = pd.read_csv("../articles.csv")
    article_preprocessor = ArticlePreprocessor(news_df)
    article_preprocessor.fully_preprocess()


if __name__ == '__main__':
    main()