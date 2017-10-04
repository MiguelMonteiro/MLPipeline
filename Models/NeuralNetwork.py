import tensorflow as tf
import numpy as np
import sklearn.preprocessing

init_learning_rate = .1


def get_logits(features, weights, biases):
    o = features
    for w, b in zip(weights, biases):
        o = tf.nn.relu(o * w + b)
    return o


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


class BatchGenerator(object):
    def __init__(self, x, y, batch_size):

        if len(y) < batch_size:
            Exception('Batch Size must be smaller or equal to the number of samples')

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.pos = 0

    def next(self):
        indices = []
        while len(indices) < self.batch_size:
            end = min(self.pos + self.batch_size, len(self.y))
            indices.append(range(self.pos, end))
        return self.x[indices], self.y[indices]


class NeuralNetwork(object):
    def __init__(self, x, y, hidden_layers, batch_size=100):
        self.x = x
        self.y = y

        self.n_features = x.shape[1]
        self.n_labels = np.unique(y)

        self.hidden_layers = hidden_layers

        self.batch_generator = BatchGenerator(x, y, batch_size)

        self.max_steps = int(1e3)

    def create_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with graph.name_scope('fit_graph') as scope:
                # Initialization
                features = tf.placeholder(tf.float64, shape=[None, self.n_features])
                labels = tf.placeholder(tf.float64, shape=[None])
                one_hot_labels = tf.one_hot(labels, self.n_labels)

                n_nodes = [self.n_features] + self.hidden_layers + [self.n_labels]
                c = .1

                weights = []
                biases = []
                for i in range(len(n_nodes) - 1):
                    weights.append(tf.Variable(tf.truncated_normal([n_nodes[i], n_nodes[i + 1]], -c, c)))
                    biases.append(tf.Variable(tf.zeros([n_nodes[i + 1]])))

                logits = get_logits(features, weights, biases)

                # Loss
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))

                # Optimizer.
                global_step = tf.Variable(0)
                learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 2000, 0.99, staircase=True)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
                # Prediction
                prediction = tf.nn.softmax(logits)

        graph.finalize()
        tf.reset_default_graph()
        return graph, weights, biases, prediction

    def create_predict_graph(self, weights, biases):
        graph = tf.Graph()
        with graph.as_default():
            with graph.name_scope('predict_graph') as scope:
                features = tf.placeholder(tf.float64, shape=[None, self.n_features])
                logits = get_logits(features, weights, biases)

                # Prediction
                prediction = tf.nn.softmax(logits)

        graph.finalize()
        tf.reset_default_graph()
        return graph, weights, biases

    def fit(self):

        graph, weights, biases, prediction = self.create_graph()

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            summary_frequency = 100
            for step in range(self.max_steps):
                x, y = self.batch_generator.next()

                feed_dict = {'features': x, 'labels': y}

                weights, biases, prediction = session.run([weights, biases, prediction], feed_dict=feed_dict)

                if step % summary_frequency == 0:
                    print('Step: %d' % step)
                    print('Minibatch accuracy: %.4f%%' % accuracy(prediction, y))


