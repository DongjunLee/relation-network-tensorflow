from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import relation_network



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.embedding_input = features["input_data"]
            self.input_mask = features["input_data_mask"]
            self.embedding_question = features["question_data"]

        self.targets = labels

    def build_graph(self):
        graph = relation_network.Graph(self.mode)
        output = graph.build(inputs=self.inputs)

        self._build_prediction(output)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(output)
            self._build_optimizer()
            self._build_metric()

    def _build_prediction(self, output):
        self.predictions = tf.argmax(output, axis=1)

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    self.targets,
                    logits,
                    scope="cross-entropy")
            reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss = tf.add(cross_entropy, reg_term)

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")

    def _build_metric(self):
        self.metrics = {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions)
        }
