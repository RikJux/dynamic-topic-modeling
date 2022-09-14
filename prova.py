from corpus import Corpus
import os
from utils import create_paths
data_files = {"trump_tweets": "tweets_01-08-2021.json", "gone_wind": "Gone with the Wind.mhtml"}
data_path = "data"
first_level_path = ["save", "results"]
types = ["trump_tweets", "gone_wind"]
all_num_topics = [5, 10, 20, 40]
n_words_topic = 30

paths_dict = create_paths(data_files, "data", first_level_path, types)
corpus_path = os.path.join(paths_dict["trump_tweets"+"_save"], "trump_tweets"+"_corpus")
corpus = Corpus.load(corpus_path)
bow, slice, dictionary = corpus()
print(len(bow))