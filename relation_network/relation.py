
import itertools
import tensorflow as tf



class RN:

    def __init__(self,
                 g_units=[256, 256, 256, 256],
                 f_units=[256, 512, 159]):

        self.g_units = g_units
        self.f_units = f_units

    def build(self, objects, question):
        object_pair_with_qs = self._combinations(objects, question)
        relations = [self._g_layer(pair) for pair in object_pair_with_qs]
        output = self._f_layer(sum(relations))
        return output

    def _combinations(self, objects, question):
        object_pair_with_q = []
        for pair in itertools.combinations(objects, 2):
            object_pair_with_q.append(tf.concat([pair[0], pair[1], question], axis=1))
        return object_pair_with_q

    def _g_layer(self, input, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("g-layer", reuse=reuse):
            hidden = tf.layers.dense(input, self.g_units[0],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="mlp-0")
            for index, unit in enumerate(self.g_units[1:]):
                hidden = tf.layers.dense(hidden, unit,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name=f"mlp-{(index+1)}")
            return hidden

    def _f_layer(self, input):
        with tf.variable_scope("f-layer"):
            hidden = tf.layers.dense(input, self.f_units[0],
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            for unit in self.f_units[1:]:
                hidden = tf.layers.dense(hidden, unit,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        return hidden
