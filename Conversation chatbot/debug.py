import tensorflow as tf

import numpy as np
import pickle
from utils import text_prepare

#pickle.dump( favorite_color, open( "save.p", "wb" ) )
word_embeddings = pickle.load(open( "word_embeddings.p", "rb" ) )
word2id = pickle.load(open( "word2id.p", "rb" ) )
id2word = pickle.load(open( "id2word.p", "rb" ) )

start_symbol_id = word2id["[^]"]
class Seq2SeqModel(object):

    def __init__(self, vocab_size, embeddings_size, hidden_size,
                 max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):


        self.declare_placeholders()
        self.create_embeddings(vocab_size, embeddings_size)
        self.build_encoder(hidden_size)
        self.build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

        # Compute loss and back-propagate.
        self.compute_loss()
        self.perform_optimization()

        # Get predictions for evaluation.
        self.train_predictions = self.train_outputs.sample_id
        self.infer_predictions = self.infer_outputs.sample_id

    def declare_placeholders(self):
        """Specifies placeholders for the model."""

        # Placeholders for input and its actual lengths.
        self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
        self.input_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_batch_lengths')

        # Placeholders for groundtruth and its actual lengths.
        self.ground_truth = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ground_thruth')
        self.ground_truth_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='ground_thrugth_length')

        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
        self.learning_rate_ph = tf.placeholder_with_default(tf.cast(0.001, tf.float32), shape=[])

    def create_embeddings(self, vocab_size, embeddings_size):
        """Specifies embeddings layer and embeds an input batch."""

        random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
        self.embeddings = tf.Variable(initial_value=random_initializer, dtype=tf.float32, name='embeding_matrix')

        # Perform embeddings lookup for self.input_batch.
        self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_batch)

    def build_encoder(self, hidden_size):
        """Specifies encoder architecture and computes its output."""

        # Create GRUCell with dropout.
        encoder_cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
        encoder_cell_dropout = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=self.dropout_ph)

        # Create RNN with the predefined cell.
        _, self.final_encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell_dropout,
                                                        inputs=self.input_batch_embedded,
                                                        sequence_length=self.input_batch_lengths,
                                                        dtype=tf.float32)

    def build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
        """Specifies decoder architecture and computes the output.

            Uses different helpers:
              - for train: feeding ground truth
              - for inference: feeding generated output

            As a result, self.train_outputs and self.infer_outputs are created.
            Each of them contains two fields:
              rnn_output (predicted logits)
              sample_id (predictions).

        """

        # Use start symbols as the decoder inputs at the first time step.
        batch_size = tf.shape(self.input_batch)[0]
        start_tokens = tf.fill([batch_size], start_symbol_id)
        ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

        # Use the embedding layer defined before to lookup embedings for ground_truth_as_input.
        self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings, ground_truth_as_input)

        # Create TrainingHelper for the train stage.
        train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded,
                                                         self.ground_truth_lengths)

        # Create GreedyEmbeddingHelper for the inference stage.
        # You should provide the embedding layer, start_tokens and index of the end symbol.
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_symbol_id)

        def decode(helper, scope, reuse=None):
            """Creates decoder and return the results of the decoding with a given helper."""

            with tf.variable_scope(scope, reuse=reuse):
                # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
                decoder_cell = tf.contrib.rnn.GRUCell(num_units=hidden_size, reuse=reuse)
                decoder_cell_dropout = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=self.dropout_ph)

                # Create a projection wrapper.
                decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell_dropout, vocab_size, reuse=reuse)

                # Create BasicDecoder, pass the defined cell, a helper, and initial state.
                # The initial state should be equal to the final state of the encoder!
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                          initial_state=self.final_encoder_state)

                # The first returning argument of dynamic_decode contains two fields:
                #   rnn_output (predicted logits)
                #   sample_id (predictions)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter,
                                                                  output_time_major=False, impute_finished=True)

                return outputs

        self.train_outputs = decode(train_helper, 'decode')
        self.infer_outputs = decode(infer_helper, 'decode', reuse=True)

    def compute_loss(self):
        """Computes sequence loss (masked cross-entopy loss with logits)."""

        weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.train_outputs.rnn_output,
            targets=self.ground_truth,
            weights=weights)

    def perform_optimization(self):
        """Specifies train_op that optimizes self.loss."""

        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self.learning_rate_ph,
            optimizer='Adam',
            clip_gradients=1.0
        )

    def get_response(self, session, input_sentence):
        print(input_sentence,"--------->")
        sentence = text_prepare(input_sentence)
        X = []
        row = []
        for word in sentence:
            if word in word2id:
                row.append(word2id[word])
            else:
                row.append(start_symbol_id)
        X.append(row)
        X = np.array(X)

        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: np.array([len(input_sentence)]),
        }
        pred = session.run([self.infer_predictions], feed_dict=feed_dict)
        return " ".join([id2word[index] for index in pred[0][0][:-1]])

    def get_reply(self, session, input_sentence):
        input_sentence = text_prepare(input_sentence)
        X = [[word2id[word] if word in word2id else start_symbol_id for word in input_sentence]]
        X = np.array(X)
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: np.array([len(input_sentence)]),
            self.ground_truth_lengths: np.array([15])
        }
        pred = session.run([self.infer_predictions], feed_dict=feed_dict)
        return " ".join([id2word[index] for index in pred[0][0][:-1]])

    def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, learning_rate, dropout_keep_probability):
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_probability
        }
        pred, loss, _ = session.run([
            self.train_predictions,
            self.loss,
            self.train_op], feed_dict=feed_dict)
        return pred, loss

    def predict_for_batch(self, session, X, X_seq_len):
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len
        }
        pred = session.run([
            self.infer_predictions
        ], feed_dict=feed_dict)[0]
        return pred

    def predict_for_batch_with_loss(self, session, X, X_seq_len, Y, Y_seq_len):
        feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len
        }
        pred, loss = session.run([
            self.infer_predictions,
            self.loss,
        ], feed_dict=feed_dict)
        return pred, loss

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


model = Seq2SeqModel(vocab_size = len(word2id),
                    embeddings_size = 300,
                    hidden_size = 128,
                    max_iter = 20,
                    start_symbol_id=word2id['[^]'],
                    end_symbol_id=word2id['[$]'],
                    padding_symbol_id=word2id['[#]'])


saver = tf.train.Saver()

for i in range (834, 845):
    saver.restore(sess, 'checkpoints/model_four_'+str(i))

    print('---------------EPOCH-------------------------'+str(i))

    print(model.get_response(sess, "hello"))
    print(model.get_response(sess, "Hi"))
    print(model.get_response(sess, "How are you?"))
    print(model.get_response(sess, "What's your name?"))
    print(model.get_response(sess, "Tell me about yourself"))
    print(model.get_response(sess, "Do you love me?"))
    print(model.get_response(sess, "What's the meaning of life?"))
    print(model.get_response(sess, "How is the weather today?"))
    print(model.get_response(sess, "Let's have a dinner"))
    print(model.get_response(sess, "Are you a bot?"))


    print("--------------------\n\n\n")