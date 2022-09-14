import pandas as pd
import os
import itertools
from bs4 import BeautifulSoup
from numpy import cumsum, argmax
from collections import Counter
from gensim.models import CoherenceModel

def open_trump_file(data_path):
    """
    returns the pair:
        list of tweets, ordered by year
        time slice (tweet count for each year), also output as csv in folder 'results'
    """

    df = pd.read_json(data_path, 'record')
    df = df.drop(columns=["retweets", "id", "device", "favorites", "isRetweet", "isDeleted", "isFlagged"])
    df["date"] = df["date"].apply(lambda s: str(s).split("-")[0])
    df = df.sort_values("date")
    slices = df.groupby("date").count()

    return list(df["text"]), list(slices["text"])

def open_gone_wind_file(data_path):
    """
    returns the pair:
        list of paragraphs
        time slice (count of paragraphs in each part), also output as csv in folder 'results'
    """

    with open(data_path, encoding='utf8') as file:
        soup = BeautifulSoup(file, "html.parser")

    curr_el = soup.find("h1", text="PART ONE")

    slices = []
    slice_counter = 0

    text = []

    while True:
        curr_el = curr_el.find_next(text=True, recursive=False)
        tag = curr_el.name
        content = curr_el.text

        if tag == "p":
            slice_counter += 1
            text.append(content)
        elif tag == "h2":
            if content == "THE END":
                slices.append(slice_counter)
                break
        elif tag == "h1":
            slices.append(slice_counter)
            slice_counter = 0
        else:
            pass

    return text, slices

def create_paths(files, data_dir, first_level_dirs, second_level_dirs):

    paths = []
    combinations = list(itertools.product(first_level_dirs, second_level_dirs))

    for comb in combinations:
        path = os.path.join(*comb)
        path_name = "_".join(reversed(comb))
        paths.append((path_name, path))
        if not os.path.isdir(path):
            os.makedirs(path)

    paths.append(("trump_tweets_file", os.path.join(data_dir, files["trump_tweets"])))
    paths.append(("gone_wind_file", os.path.join(data_dir, files["gone_wind"])))

    return dict(paths)


def prepare_df_trump_tweets(seq_lda, n_words_topic):
    """
    prepare resulting topics to be stored in a pandas dataframe
    """
    tuples = []
    for i in range(seq_lda.num_topics):
        topic_times = seq_lda.print_topic_times(i, n_words_topic)
        for topic_year in range(len(topic_times)):
            year = topic_year + 2009
            for tuple in topic_times[topic_year]:
                tuples.append((i, year, *tuple))

    df = pd.DataFrame(tuples, columns=["topic", "year", "word", "relevance"])
    df["rank"] = list(range(n_words_topic)) * len(seq_lda.time_slice) * seq_lda.num_topics
    df = pd.pivot(df, values=["word", "relevance"], columns=["topic", "year"], index=["rank"])

    return df

def prepare_df_gone_wind(seq_lda, n_words_topic):
    tuples = []
    parts = ["PART ONE", "PART TWO", "PART THREE", "PART FOUR", "PART FIVE"]
    for i in range(seq_lda.num_topics):
        topic_times = seq_lda.print_topic_times(i, n_words_topic)
        for topic_part in range(len(topic_times)):
            part = parts[topic_part]
            for tuple in topic_times[topic_part]:
                tuples.append((i, part, *tuple))

    df = pd.DataFrame(tuples, columns=["topic", "part", "word", "relevance"])
    df["rank"] = list(range(n_words_topic)) * len(seq_lda.time_slice) * seq_lda.num_topics
    df = pd.pivot(df, values=["word", "relevance"], columns=["topic", "part"], index=["rank"])

    return df

def prepare_topic_pop_df_trump_tweets(seq_lda):
    slices = cumsum([0] + seq_lda.time_slice)
    cuts = []
    for i in range(1, len(slices)):
        cuts.append((i+2008, (slices[i-1], slices[i])))

    dict_el = []
    for time_label, rng in cuts:
        counter_res = Counter([argmax(seq_lda.doc_topics(doc_i)) for doc_i in range(*rng)])
        dict_el.append((time_label, counter_res))
        
    return pd.DataFrame(dict(dict_el)).T.sort_index(axis=1)

def prepare_topic_pop_df_gone_wind(seq_lda):
    slices = cumsum([0] + seq_lda.time_slice)
    cuts = []
    parts = ["PART 1", "PART 2", "PART 3", "PART 4", "PART 5"]
    for i in range(1, len(slices)):
        cuts.append((parts[i-1], (slices[i-1], slices[i])))

    dict_el = []
    for time_label, rng in cuts:
        counter_res = Counter([argmax(seq_lda.doc_topics(doc_i)) for doc_i in range(*rng)])
        dict_el.append((time_label, counter_res))
        
    return pd.DataFrame(dict(dict_el)).T.sort_index(axis=1)

def coherence_over_time(seq_lda, corpus, type_c):

    cuts = []
    docs, slices, _ = corpus()
    slices = cumsum([0] + slices)

    if type_c == "trump_tweets":
       cuts = [i+2009 for i in range(len(slices))] 
       cuts_name = "year"
    else:
        cuts = ["PART 1", "PART 2", "PART 3", "PART 4", "PART 5"]
        cuts_name = "part"

    coherence = []
    for i in range(len(slices)-1):
        coh_model = CoherenceModel.for_topics(seq_lda.dtm_coherence(i), corpus=docs[slices[i]:slices[i+1]], dictionary=seq_lda.id2word, coherence='u_mass')
        coherence.append((cuts[i], coh_model.get_coherence(), seq_lda.num_topics))

    df = pd.DataFrame(coherence, columns=[cuts_name, "coherence", "num_topics"])
    df = pd.pivot(df, values=["coherence"], columns=[cuts_name], index=["num_topics"])

    return df
