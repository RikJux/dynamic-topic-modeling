import utils
import pickle
from pipeline import Pipeline

class Corpus:
    """
    This class contains all the data required by the gensim LdaSeqModel
    Also, it's responsible for calling methods to parse data
    """
    
    def __init__(self, data_path, type) -> None:

        raw_corpus, self._time_slice = utils.open_trump_file(data_path) if type=="trump_tweets" else utils.open_gone_wind_file(data_path)
        self._pipeline = Pipeline(raw_corpus, type)

    
    def __call__(self):
        """
        Returns all the data required by gensim model (LdaSeqModel)
        """
        bow, dict = self._pipeline()
        return bow, self._time_slice, dict

    def save(self, save_to):
        pickle.dump(self, open(save_to, 'wb'))

    def load(load_from):
        return pickle.load(open(load_from, 'rb'))
