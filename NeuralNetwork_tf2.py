import tensorflow as tf
import numpy as np
from Enums import ActivationFunction
from helper_functions.helper_functions import *
import tensorflow_probability as tfp

tf.compat.v1.disable_v2_behavior()


# Neural network class
class NeuralNetwork:
    def __init__(self, inputDimension, hiddenLayers, activationFunction=ActivationFunction.Tanh, seed=None):
        self.layers = [inputDimension] + hiddenLayers + [1]
        self.activationFunction = activationFunction

        if seed is not None:
            tf.compat.v1.set_random_seed(seed)
            np.random.seed(seed)

        # Network parameters
        self.weights, self.biases = self.CreateNetworkParameters()

        # Input placeholders
        self.xInt = []
        self.xBound = []
        self.xIntValidate = []
        self.xBoundValidate = []
        for i in range(inputDimension):
            self.xInt.append(tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="xInt" + str(i)))
            self.xBound.append(tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="xBound" + str(i)))
            self.xIntValidate.append(tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="xIntValidate" + str(i)))
            self.xBoundValidate.append(tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="xBoundValidate" + str(i)))

        # Outputs
        self.yInt = self.CreateGraph(self.xInt, activationFunction)
        self.yBound = self.CreateGraph(self.xBound, activationFunction)
        self.yIntValidate = self.CreateGraph(self.xIntValidate, activationFunction)
        self.yBoundValidate = self.CreateGraph(self.xBoundValidate, activationFunction)

        # Boundary condition & Source functions
        self.boundaryCondition = tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="BoundaryCondition")
        self.boundaryConditionValidate = tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="BoundaryConditionValidate")
        self.sourceFunction = tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="SourceFunction")
        self.sourceFunctionValidate = tf.compat.v1.placeholder(tf.float64, shape=[None, 1], name="SourceFunctionValidate")

        self.saver = tf.compat.v1.train.Saver()

        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())

    def CreateNetworkParameters(self):
        weights = []
        biases = []

        for i in range(len(self.layers) - 1):
            w = tf.Variable(NeuralNetwork.GlorotInitializer(self.layers[i], self.layers[i + 1]), dtype=tf.float64)
            b = tf.Variable(tf.zeros([1, self.layers[i + 1]], dtype=tf.float64), dtype=tf.float64)

            weights.append(w)
            biases.append(b)

        return weights, biases

    @staticmethod
    def GlorotInitializer(dim1, dim2):
        return tf.random.uniform([dim1, dim2],
                                 minval=- np.sqrt(6 / (dim1 + dim2)),
                                 maxval=np.sqrt(6 / (dim1 + dim2)), dtype=tf.float64)

    def CreateGraph(self, x, activationFunction):
        y = tf.concat(x, axis=1)
        for i in range(len(self.layers) - 2):
            w = self.weights[i]
            b = self.biases[i]

            if activationFunction == ActivationFunction.Tanh:
                y = tf.nn.tanh(tf.add(tf.matmul(y, w), b))

            elif activationFunction == ActivationFunction.Sigmoid:
                y = tf.nn.sigmoid(tf.add(tf.matmul(y, w), b))

            elif activationFunction == ActivationFunction.Sin:
                y = tf.sin(tf.add(tf.matmul(y, w), b))

            elif activationFunction == ActivationFunction.Cos:
                y = tf.cos(tf.add(tf.matmul(y, w), b))

            elif activationFunction == ActivationFunction.Atan:
                y = tf.atan(tf.add(tf.matmul(y, w), b))

        w = self.weights[-1]
        b = self.biases[-1]
        return tf.add(tf.matmul(y, w), b)

    @make_val_and_grad_fn
    def loss_func_val_grad(lossFunction):
        return tf.math.log(lossFunction)
    
    def start_lbfgs_minimize(lossFunction):
        return np.random.randn(tf.math.log(lossFunction))
    
    @tf.function
    def Train(self, lossFunction, iterations, feedDict, fetchList, callback):
#         optimizer = tf.contrib.opt.ScipyOptimizerInterface(tf.log(lossFunction),
#                                                            method='L-BFGS-B',
#                                                            options={'maxiter': iterations,
#                                                                     'maxfun': iterations,
#                                                                     'maxcor': 50,
#                                                                     'maxls': 50,
#                                                                     'ftol': 1.0 * np.finfo(np.float64).eps,
#                                                                     'gtol': 0.000001})

#         optimizer.minimize(self.session, feed_dict=feedDict, fetches=fetchList, loss_callback=callback)
        
        
        
        tfp.optimizer.lbfgs_minimize(loss_func_val_grad,
                                     initial_position=tf.constant(start_lbfgs_minimize(lossFunction)),
                                     max_iterations=iterations)

    def Predict(self, x):
        feedDict = dict()
        for i in range(len(x)):
            feedDict[self.xInt[i]] = x[i]
        return self.session.run(self.yInt, feed_dict=feedDict)

    def SaveWeights(self, path="Autosave"):
        self.saver.save(self.session, path)

    def LoadWeights(self, path="Autosave"):
        self.saver.restore(self.session, path)

    def Cleanup(self):
        self.session.close()
        tf.compat.v1.reset_default_graph()
