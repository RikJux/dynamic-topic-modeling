from nltk.tokenize import TweetTokenizer, TreebankWordTokenizer
from spacy.tokens import Doc
from string import punctuation
import itertools

class Tokenizer:

    def __init__(self, nlp, type) -> None:
        self.vocab = nlp.vocab
        self._stopw = nlp.Defaults.stop_words
        self._type = type
        self._tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True) if self._type == "trump_tweets" else TreebankWordTokenizer()

    def __call__(self, text):
        
        if self._type == "trump_tweets":
            tokens = [word.replace("'s", "") for word in self._tokenizer.tokenize(text) 
                        if word not in punctuation and 
                        word not in self._stopw and 
                        not word.startswith(('http', '#', "&", ".")) and
                        len(word) > 1 and
                        word not in ["rt", "amp", ":/"]]
        else:
            tokens = []
            for word in self._tokenizer.tokenize(text):
                tokens.append(word.split("=97"))
            tokens = list(itertools.chain(*tokens))
            tokens = [word.replace("'s", "").rstrip(".").lower() for word in tokens
                        if word not in punctuation and 
                        word not in self._stopw and
                        len(word) > 1]
            tokens = [word for word in tokens if len(word) > 2 and word not in self._stopw]

        return Doc(self.vocab, words=tokens, spaces=[True] * len(tokens))

