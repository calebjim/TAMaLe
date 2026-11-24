import tensorflow as tf

class ClassRecall(tf.keras.metrics.Metric):
    def __init__(self, class_id, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_hat = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        is_true = tf.equal(y_true, self.class_id)
        is_pred = tf.equal(y_hat, self.class_id)

        tp = tf.cast(tf.logical_and(is_true, is_pred), self.dtype)
        fn = tf.cast(tf.logical_and(is_true, tf.logical_not(is_pred)), self.dtype)

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
            tp, fn = tp*sw, fn*sw

        self.true_positives.assign_add(tf.reduce_sum(tp))
        self.false_negatives.assign_add(tf.reduce_sum(fn))

    def result(self):
        return tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)


class ClassF1(tf.keras.metrics.Metric):
    def __init__(self, class_id, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_hat = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        is_true = tf.equal(y_true, self.class_id)
        is_pred = tf.equal(y_hat, self.class_id)

        tp = tf.cast(tf.logical_and(is_true, is_pred), self.dtype)
        fn = tf.cast(tf.logical_and(is_true, tf.logical_not(is_pred)), self.dtype)
        fp = tf.cast(tf.logical_and(tf.logical_not(is_true), is_pred), self.dtype)

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
            tp, fn, fp = tp*sw, fn*sw, fp*sw

        self.true_positives.assign_add(tf.reduce_sum(tp))
        self.false_negatives.assign_add(tf.reduce_sum(fn))
        self.false_positives.assign_add(tf.reduce_sum(fp))

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives+self.false_positives)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives+self.false_negatives)
        return tf.math.divide_no_nan(2 * precision * recall, precision + recall)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)
        self.false_positives.assign(0.0)
