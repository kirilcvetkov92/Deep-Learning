import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

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

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        self.chatbot = ChatBot(
            'ckorpion',
            trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
        )

        self.chatbot.set_trainer(ListTrainer)
        self.chatbot.train([
            "Hi",
            "Hello. I can help you with stack overflow queries",
        ])

        self.chatbot.train([
            "Hey",
            "Hello. I can help you with stack overflow queries :)",
        ])


        self.chatbot.train([
            "Who is the oldest human being on earth?",
            "Olga Paleolitkata :)",
        ])

        self.chatbot.train([
            "What's your name?",
            "My name is CKorpion nicknamed by Cvetkov Kiril :). \n What's your name?",
        ])

        self.chatbot.train([
            "Martin",
            "Did you mean embe :D ",
        ])

        self.chatbot.train("chatterbot.corpus.english")



    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent[0] == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(prepared_question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]

            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)

            return self.ANSWER_TEMPLATE % (tag, thread_id)

