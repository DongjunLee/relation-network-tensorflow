
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
              input_mask=None,
              question=None):
        embedding_input, embedding_question = self._build_embed(input, question)
        facts, question = self._build_input_module(embedding_input, input_mask, embedding_question)
        output = self._build_relational_module(facts, question)
        return output

    def _build_embed(self, input, question):
        with tf.variable_scope ("embeddings", dtype=self.dtype) as scope:

            embedding = tf.get_variable(
                    "word_embedding", [Config.data.vocab_size, Config.model.embed_dim], self.dtype)
            embedding_input = tf.nn.embedding_lookup(embedding, input)
            embedding_question = tf.nn.embedding_lookup(embedding, question)
            return embedding_input, embedding_question

    def _build_input_module(self, embedding_input, input_mask, embedding_question):
        encoder = Encoder(
            encoder_type=Config.model.encoder_type,
            num_layers=Config.model.num_layers,
            cell_type=Config.model.cell_type,
            num_units=Config.model.num_units,
            dropout=Config.model.dropout)

        # slice zeros padding
        input_length = tf.reduce_max(input_mask, axis=1)
        question_length = tf.reduce_sum(tf.to_int32(
            tf.not_equal(tf.reduce_max(embedding_question, axis=2), Config.data.PAD_ID)), axis=1)

        with tf.variable_scope("input-module") as scope:
            input_encoder_outputs, _ = encoder.build(
                    embedding_input, input_length, scope="encoder")

            with tf.variable_scope("facts") as scope:
                batch_size = tf.shape(input_mask)[0]
                max_mask_length = tf.shape(input_mask)[1]

                def get_encoded_fact(i):
                    nonlocal input_mask

                    mask_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_mask[i], Config.data.PAD_ID)), axis=0)
                    input_mask = tf.boolean_mask(input_mask[i], tf.sequence_mask(mask_lengths, max_mask_length))

                    encoded_facts = tf.gather_nd(input_encoder_outputs[i], tf.reshape(input_mask, [-1, 1]))
                    zero_padding = tf.zeros(tf.stack([max_mask_length - mask_lengths, Config.model.num_units]))
                    return tf.concat([encoded_facts, zero_padding], axis=0)

                facts_stacked = tf.map_fn(get_encoded_fact, tf.range(start=0, limit=batch_size), dtype=self.dtype)

                # max_input_mask_length x [batch_size, num_units]
                facts = tf.unstack(tf.transpose(facts_stacked, [1, 0, 2]), num=Config.data.max_input_mask_length)

        with tf.variable_scope("input-module", reuse=True):
            _, question = encoder.build(
                    embedding_question, question_length, scope="encoder")

        return facts, question

    def _build_relational_module(self, facts, question):
        with tf.variable_scope("relational-network-module"):
            rn = RN(g_units=Config.model.g_units,
                    f_units=Config.model.f_units)
            return rn.build(facts, question)
