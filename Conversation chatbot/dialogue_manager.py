import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatbot import *

from utils import *
import tensorflow as tf

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

        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)

        best_thread = pairwise_distances_argmin(
            X=question_vec.reshape(1, -1),
            Y=thread_embeddings,
            metric='cosine'
        )
        return thread_ids[best_thread[0]]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        self.create_chitchat_bot()


    def create_chitchat_bot(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.model = Seq2SeqModel(vocab_size=len(word2id),
                             embeddings_size=300,
                             hidden_size=128,
                             max_iter=20,
                             start_symbol_id=word2id['[^]'],
                             end_symbol_id=word2id['[$]'],
                             padding_symbol_id=word2id['[#]'])

        saver = tf.train.Saver()

        saver.restore(self.sess, 'checkpoints/model_four_691')



    def generate_answer(self, question):
        # Pass question to chitchat_bot to generate a response.
        response = self.model.get_response(self.sess, question)
        return response



