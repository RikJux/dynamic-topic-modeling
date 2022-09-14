import pandas as pd
from pandas.plotting import table 
import pickle
from utils import prepare_df_trump_tweets, prepare_df_gone_wind
from utils import prepare_topic_pop_df_trump_tweets, prepare_topic_pop_df_gone_wind
from utils import coherence_over_time
import matplotlib.pyplot as plt

class Results:
    """
    wrapper for the pandas dataframe holding the meaningful results
    also wraps calls to the matplotlib library
    """
    def __init__(self, seq_lda, corpus, n_words_topic, type_c) -> None:
        if type_c == "trump_tweets":
            df = prepare_df_trump_tweets(seq_lda, n_words_topic)
            df_pop = prepare_topic_pop_df_trump_tweets(seq_lda)
        else:
            df = prepare_df_gone_wind(seq_lda, n_words_topic)
            df_pop = prepare_topic_pop_df_gone_wind(seq_lda)

        self._topics_word_df = df
        self._topics_pop_df = df_pop
        self._coherence_over_time = coherence_over_time(seq_lda, corpus, type_c)

        self._n_words_topic = n_words_topic

    def save(self, save_to):
        pickle.dump(self, open(save_to, 'wb'))

    def load(load_from):
        return pickle.load(open(load_from, 'rb'))

    def get_coherence(self):
        return self._coherence_over_time

    def plot_cum_pop(self, save_path):

        fig, ax = plt.subplots()
        self._topics_pop_df.cumsum().plot(ax=ax)
        ax.legend(prop={'size':16}, loc="upper left")
        fig.savefig(save_path)

    def plot_topic_evolution(self, num_topic, save_path, num_words=0):

        if num_words < 1:
            num_words = self._n_words_topic

        topic_df = self._show_topic_evolution(num_topic, num_words)
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=topic_df.values, colLabels=topic_df.columns, loc='center',
                        cellLoc="center",
                        edges="vertical")
        fig.tight_layout()
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 0.5)
        fig.savefig(save_path)

    def plot_word_evolution(self, num_topic, words_list, save_path):

        fig, ax = plt.subplots()
        ax.plot(self._show_word_evolution(num_topic, words_list), marker='o', label=words_list)
        ax.legend(prop={'size':16})
        plt.title("Relevance of words in topic " + str(num_topic), fontsize=16)
        fig.savefig(save_path)

    def _show_topic_evolution(self, num_topic, num_words):

        df = self._topics_word_df["word"][num_topic]
        return df[df.index < num_words]

    def _show_word_evolution(self, num_topic, words_list):

        dfs = []
        for word in words_list:
            dfs.append(self._show_single_word_evolution(num_topic, word))

        return pd.concat(dfs, axis=1)

    def _show_single_word_evolution(self, num_topic, word):
        mask_df = self._topics_word_df["word"][num_topic].applymap(lambda x: x==word)
        rel_df = self._topics_word_df["relevance"][num_topic].where(mask_df, 0).sum(axis=0)
        return rel_df

    def get_best_num_topics(list_of_results):
        df = Results._concat_all(list_of_results)

        fig, ax = plt.subplots()
        df.plot(ax=ax)
        plt.show()

        return df.mean().idxmax()

    def _concat_all(list_of_results):
        all_coherence = [r.coherence_over_time["coherence"] for r in list_of_results]
        df = pd.concat(all_coherence, axis=0).T

        return df