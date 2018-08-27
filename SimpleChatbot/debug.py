import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim+21)

        best_thread = pairwise_distances_argmin(
            X=question_vec.reshape(1,-1),
            Y=thread_embeddings,
            metric='cosine'
        )
        return thread_ids[best_thread[0]]


t = ThreadRanker(RESOURCE_PATH)
t = t.get_best_thread('what is c++', 'c_cpp')
print(t)