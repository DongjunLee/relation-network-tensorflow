
from hbconfig import Config
import tensorflow as tf

from .encoder import Encoder



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self,
              embedding_input=None,
              input_mask=None,
              embedding_question=None):
        facts, question = self._build_input_module(embedding_input, input_mask, embedding_question)
        output = self._build_relational_module(facts, question)
        return output

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
                    padding = tf.zeros(tf.stack([max_mask_length - mask_lengths, Config.model.num_units]))
                    return tf.concat([encoded_facts, padding], 0)

                facts_stacked = tf.map_fn(get_encoded_fact, tf.range(start=0, limit=batch_size), dtype=self.dtype)

                # max_input_mask_length x [batch_size, num_units]
                facts = tf.unstack(tf.transpose(facts_stacked, [1, 0, 2]), num=Config.data.max_input_mask_length)

        with tf.variable_scope("input-module") as scope:
            scope.reuse_variables()
            _, question = encoder.build(
                    embedding_question, question_length, scope="encoder")

        return facts, question[0]

    def _build_relational_module(self, facts, question):
        # TODO:
        # 1. Object(fact) pair with question
        # 2. g_layer (MLP) -> element-wise sum
        # 3. f_layer (MLP) -> output
        pass
