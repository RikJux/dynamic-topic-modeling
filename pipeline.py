import spacy
from spacy.tokens import Doc
from spacy.language import Language
from gensim.corpora import Dictionary
from tokenizer import Tokenizer

class Pipeline:
    """
    wrapper for all the calls to the spacy library
    prepares data for the model
    """

    def __init__(self, docs, type, model="en_core_web_sm") -> None:
        self._dictionary = Dictionary()

        @Language.component("lemma")
        def lemma_function(doc):

            doc._.lemmas = [word.lemma_ for word in doc]

            return doc

        @Language.component("words_in_dict")
        def words_in_dict_function(doc): # as_tuples = True

            self._dictionary.add_documents([doc._.lemmas])

            return doc

        @Language.component("bow")
        def bow_function(doc):

            doc._.bow = self._dictionary.doc2bow(doc._.lemmas)

            return doc

        nlp = spacy.load(model)

        if not Doc.has_extension("id"):
            Doc.set_extension("id", default=None)
        if not Doc.has_extension("lemmas"):
            Doc.set_extension("lemmas", default=[])
        if not Doc.has_extension("bow"):
            Doc.set_extension("bow", default=[])

        nlp.tokenizer = Tokenizer(nlp, type)
        nlp.add_pipe("lemma", last=True)
        nlp.add_pipe("words_in_dict", last=True)
        nlp.add_pipe("bow", last=True)

        self._bow = [doc._.bow for doc in nlp.pipe(docs)]

    
    def __call__(self):
        return self._bow, self._dictionary