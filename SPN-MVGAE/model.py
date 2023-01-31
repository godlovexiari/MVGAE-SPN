from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf
import region_graph
import RAT_SPN
flags = tf.app.flags
FLAGS = flags.FLAGS
import numpy as np
np.random.seed(17)
tf.set_random_seed(17)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)

class CSPNModelAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, y_dims, num_sum_weights, num_leaf_weights, **kwargs):
        super(CSPNModelAE, self).__init__(**kwargs)
        self.y_dims = y_dims
        self.output_shape = (2708, y_dims)
        self.output_shape = list(self.output_shape)
        self.output_dims = 1
        for dim in self.output_shape[1:]:
            self.output_dims *= int(dim)
        self.inputs1 = placeholders['features']
        self.y_ph =  placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = 2708
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.num_sum_weights = num_sum_weights
        self.num_leaf_weights = num_leaf_weights
        self.build()

    def _build(self):

        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,            
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs1)

        self.hidden11 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.h1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs1)

        self.hidden12 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.h3,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs1)
        self.hidden13 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.h5,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs1)

        self.hidden14 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.h7,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs1)
                                              
        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)
        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z_mean1 = GraphConvolution(input_dim=FLAGS.h1,
                                       output_dim=FLAGS.h2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden11)
        self.z_log_std1 = GraphConvolution(input_dim=FLAGS.h1,
                                          output_dim=FLAGS.h2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden11)

        self.z_mean2 = GraphConvolution(input_dim=FLAGS.h3,
                                       output_dim=FLAGS.h4,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden12)
        self.z_log_std2 = GraphConvolution(input_dim=FLAGS.h3,
                                          output_dim=FLAGS.h4,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden12)

        self.z_mean3 = GraphConvolution(input_dim=FLAGS.h5,
                                       output_dim=FLAGS.h6,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden13)
        self.z_log_std3 = GraphConvolution(input_dim=FLAGS.h5,
                                          output_dim=FLAGS.h6,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden13)

        self.z_mean4 = GraphConvolution(input_dim=FLAGS.h7,
                                       output_dim=FLAGS.h8,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden14)
        self.z_log_std4 = GraphConvolution(input_dim=FLAGS.h7,
                                          output_dim=FLAGS.h8,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden14)
                                          
                                         
        print("self.z_mean", self.z_mean.shape)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
        self.z1 = self.z_mean1 + tf.random_normal([self.n_samples, FLAGS.h2]) * tf.exp(self.z_log_std1)
        self.z2 = self.z_mean2 + tf.random_normal([self.n_samples, FLAGS.h4]) * tf.exp(self.z_log_std2)
        self.z3 = self.z_mean3 + tf.random_normal([self.n_samples, FLAGS.h6]) * tf.exp(self.z_log_std3)
        self.z4 = self.z_mean4 + tf.random_normal([self.n_samples, FLAGS.h8]) * tf.exp(self.z_log_std4)
        self.z = (0.2*self.z + 0.2*self.z1 + 0.2*self.z2 + 0.2*self.z3 + 0.2*self.z4)# + 0.2*self.z5)
        self.z_mean = 1*(self.z_mean + self.z_mean1 + self.z_mean2 + self.z_mean3 + self.z_mean4)# + self.z_mean5)
        self.z_log_std = 1*(self.z_log_std + self.z_log_std1 + self.z_log_std2 + self.z_log_std3 + self.z_log_std4)
        
        self.sum_weights_1 = tf.layers.dense(inputs=self.z_mean,
                                             units=self.num_sum_weights*16,
                                             activation=None)
        self.sum_weights = tf.layers.dense(inputs=self.sum_weights_1,
                                           units=self.num_sum_weights,
                                           activation=None)
        self.sum_weights = tf.reshape(self.sum_weights, [self.n_samples, self.num_sum_weights])
        print("self.sum_weights", self.sum_weights)
        self.leaf1 = tf.layers.dense(inputs=self.z_mean,
                                     units=self.output_dims * 16,  
                                     activation=tf.nn.relu)
        self.leaf2 = tf.layers.dense(inputs=self.leaf1,
                                     units=self.output_dims * 16,
                                     activation=tf.nn.relu)
        self.leaf_linear = tf.reshape(self.leaf2, [self.n_samples, self.output_dims, 16])
        self.leaf_weights = tf.layers.dense(inputs=self.leaf_linear,
                                            units=self.num_leaf_weights,
                                            activation=None)
        print("self.leaf_weights", self.leaf_weights)


        self.param_provider = RAT_SPN.ScopeBasedParamProvider(self.sum_weights, self.leaf_weights, 'encoder')   

        rg = region_graph.RegionGraph(range(self.y_dims))
        for _ in range(0, 2):
            rg.random_split(2, 1)

        args = RAT_SPN.SpnArgs()
        args.normalized_sums = True
        args.param_provider = self.param_provider
        args.num_sums = 5
        args.num_gauss = 4 
        args.dist = 'Gauss'
        self.spn = RAT_SPN.RatSpn(1, region_graph=rg, name="encoder", args=args)

        #self.z_c = self.spn.reconstruct_batch_tf(batch_size=self.n_samples, sample=True)   
        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)  
