
from hbconfig import Config
import tensorflow as tf

from .encoder import Encoder
from .relation import RN



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self,
              input=None,
              question=None):

        input_lengths = tf.reduce_sum(
                tf.to_int32(tf.not_equal(input, Config.data.PAD_ID)), axis=2,
                name="input_lengths")
        question_lengths = tf.reduce_sum(
                tf.to_int32(tf.not_equal(question, Config.data.PAD_ID)), axis=1,
                name="question_lengths")

        embedding_input, embedding_question = self._build_embed(input, question)
        facts, question = self._build_input_module(embedding_input, input_lengths,
                                                   embedding_question, question_lengths)
        output = self._build_relational_module(facts, question)
        return output

    def _build_embed(self, input, question):
        with tf.variable_scope ("embeddings", dtype=self.dtype) as scope:
            embedding = tf.get_variable(
                    "word_embedding", [Config.data.vocab_size, Config.model.embed_dim],
                    dtype=self.dtype, trainable=False)
            embedding_input = tf.nn.embedding_lookup(embedding, input)
            embedding_question = tf.nn.embedding_lookup(embedding, question)

            return embedding_input, embedding_question

    def _build_input_module(self, embedding_input, input_lengths,
                            embedding_question, question_lengths):
        encoder = Encoder(
            encoder_type=Config.model.encoder_type,
            num_layers=Config.model.num_layers,
            cell_type=Config.model.cell_type,
            num_units=Config.model.num_units,
            dropout=Config.model.dropout)

        with tf.variable_scope("input-module"):
            facts = []
            with tf.variable_scope("facts", reuse=tf.AUTO_REUSE):
                embedding_input_transpose = tf.transpose(embedding_input, [1, 0, 2, 3])
                embedding_sentences = tf.unstack(embedding_input_transpose, num=Config.data.max_fact_count)

                input_lengths_transpose = tf.transpose(input_lengths, [1, 0])
                sentence_lengths = tf.unstack(input_lengths_transpose, num=Config.data.max_fact_count)

                for embedding_sentence, sentence_length in zip(embedding_sentences, sentence_lengths):
                    _, fact = encoder.build(embedding_sentence, sentence_length, scope="fact-encoder")
                    facts.append(fact)

        with tf.variable_scope("input-module"):
            _, question = encoder.build(
                    embedding_question, question_lengths, scope="question-encoder")

        return facts, question

    def _build_relational_module(self, facts, question):
        with tf.variable_scope("relational-network-module"):
            rn = RN(g_units=Config.model.g_units,
                    f_units=Config.model.f_units + [Config.data.vocab_size])
            return rn.build(facts, question)
