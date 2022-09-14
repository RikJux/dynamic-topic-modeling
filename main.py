import pickle
from statistics import mean
from gensim.models import LdaSeqModel
from utils import create_paths
from itertools import product
from multiprocessing import Pool
import os
from corpus import Corpus
from results import Results
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

def compute_results(paths_dict, type_c, num_topics, random_seed, n_words_topic): 

    num_topics_str = str(num_topics)
    print_info = " for "+type_c+", "+num_topics_str+" "+"topics"

    result_path = os.path.join(paths_dict[type_c+"_results"], type_c+"_results_"+num_topics_str)
    if not os.path.exists(result_path): #no results previously saved
        print("generating results"+print_info)
        
        corpus_path = os.path.join(paths_dict[type_c+"_save"], type_c+"_corpus")
        corpus = Corpus.load(corpus_path)
        print("corpus loaded"+print_info)

        model_path = os.path.join(paths_dict[type_c+"_save"], type_c+"_model_"+num_topics_str)
        if not os.path.exists(model_path): #no model previously saved
            print("generating model"+print_info)

            seq_lda = LdaSeqModel(*corpus(), 
                num_topics=num_topics,
                random_state=random_seed,
                chain_variance=0.05)
            seq_lda.save(model_path)
            print("model generated"+print_info)

        else: #load the saved model
            seq_lda = LdaSeqModel.load(model_path)
            print("model loaded"+print_info)

        results = Results(seq_lda, corpus, n_words_topic, type_c)
        results.save(result_path)
        print("results generated"+print_info)

    else: #load the saved results
        results = Results.load(result_path)
        print("results loaded"+print_info)

    return (type_c, num_topics, results)


if __name__ == "__main__":

    random_seed = 101

    data_files = {"trump_tweets": "tweets_01-08-2021.json", "gone_wind": "Gone with the Wind.mhtml"}
    data_path = "data"
    first_level_path = ["save", "results"]
    types = ["trump_tweets", "gone_wind"]
    all_num_topics = [5, 10, 20]
    n_words_topic = 30

    paths_dict = create_paths(data_files, "data", first_level_path, types)

    combinations = (list(product(*[[paths_dict], types, all_num_topics, [random_seed], [n_words_topic]])))

    for type_c in types: #generate all the corpora
        corpus_path = os.path.join(paths_dict[type_c+"_save"], type_c+"_corpus")
        if not os.path.exists(corpus_path): #no corpus previously saved
            print("generating corpus for "+type_c)

            corpus = Corpus(paths_dict[type_c+"_file"], type_c)
            corpus.save(corpus_path)
            print("corpus generated for "+type_c)

    with Pool() as pool:
        all_results = pool.starmap(compute_results, combinations)

    res_dict = dict([(type_c, []) for type_c in types])

    for (type_c, num_topics, result) in all_results:
        res_dict[type_c].append((num_topics, result))

    res_dict = {k:dict(v) for k,v in res_dict.items()}

    chosen_n_topics = {"trump_tweets":20, "gone_wind":5}

    label_size = 16
    rcParams['xtick.labelsize'] = label_size 
    rcParams['ytick.labelsize'] = label_size

    for type_c in types:
        visualization_path = os.path.join(paths_dict[type_c+"_results"], "visualization")
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)

        rcParams['figure.figsize'] = 12, 10

        coherence_path = os.path.join(visualization_path, "coherence.png")
        if not os.path.exists(coherence_path):
            all_coherence = [coh._coherence_over_time for (_, coh) in res_dict[type_c].items()]
            df = pd.concat(all_coherence, axis=0)
            fig, ax = plt.subplots()
            df["coherence"].T.plot(ax=ax)
            plt.title(type_c, fontsize=label_size)
            ax.legend(prop={'size':label_size})
            fig.savefig(coherence_path)

        sum_coherence_path = os.path.join(visualization_path, "sum_coherence.png")
        if not os.path.exists(sum_coherence_path):
            all_coherence = [coh._coherence_over_time for (_, coh) in res_dict[type_c].items()]
            df = pd.concat(all_coherence, axis=0)
            fig, ax = plt.subplots()
            df["coherence"].T.mean().plot(ax=ax)
            plt.title(type_c, fontsize=label_size)
            fig.savefig(sum_coherence_path)

        popularity_path = os.path.join(visualization_path, "popularity.png")
        num_topics = chosen_n_topics[type_c]
        chosen_res = res_dict[type_c][num_topics]

        chosen_topics = ()

        if type_c == "gone_wind":
            chosen_topics = [(1, ["love", "money", "war"]), (2, ["melanie", "rhett", "ashley", "wade", "bonnie", "india"])]
        else:
            chosen_topics = [(9, ["iran", "libya", "nuclear", "ebola", "administration"]),\
                (16, ["impeachment", "clinton", "bush", "obama", "cruz", "collusion"])]

        for num_topic, word_list in chosen_topics:
            if len(word_list) > 0:
                word_evo_path = os.path.join(visualization_path, "word_evo_{n}.png".format(n=num_topic))
                if not os.path.exists(word_evo_path):
                    chosen_res.plot_word_evolution(num_topic, word_list, word_evo_path)

        if not os.path.exists(popularity_path):
            chosen_res.plot_cum_pop(popularity_path)

        rcParams['figure.figsize'] = 12, 3
        for i in range(num_topics):
            i_path = os.path.join(visualization_path, "topic_{n}.png".format(n=i))
            if not os.path.exists(i_path):
                chosen_res.plot_topic_evolution(i, i_path, 15)
