import tensorflow as tf
import numpy as np

class MultiArmedBandit():
    def __init__(self, input_dimension=[], output_dimension=[], layer_sizes=[], learning_rate=1e-4, model_ckpt=None):
        
        if model_ckpt is None:
            self.input, self.output = _construct_network(input_dimension, output_dimension, layer_sizes)

            nrof_gpu = 0 # {0, 1}

            config = tf.ConfigProto( device_count = {'GPU': nrof_gpu} )

            self.sess = tf.Session( config = config )
            #self.sess = tf.Session()

            optimizer = tf.train.AdamOptimizer(learning_rate)

            self.pt_step, self.pt_target, self.pt_loss = _setup_pre_training(self.output, output_dimension, optimizer)

            self.update_step, self.action, self.target, self.loss = _setup_training(self.output, output_dimension, optimizer)

            self.sigmoid_output = tf.sigmoid( self.output )

            self.initialize()
        else:
            
            # Load model from checkpoint saved earlier
            self.sess = tf.Session()
            
            saver = tf.train.import_meta_graph(model_ckpt + "/model.ckpt.meta")
            
            saver.restore(self.sess, tf.train.latest_checkpoint( model_ckpt ) )
            
            print("Model restored from " + model_ckpt)
            
            graph = tf.get_default_graph()
            
            self.input = graph.get_tensor_by_name("input:0")
            
            self.output = graph.get_tensor_by_name("output:0")
            
            self.action = graph.get_tensor_by_name("action:0")
            
            self.target = graph.get_tensor_by_name("target:0")
            
            self.loss = graph.get_tensor_by_name("loss:0")
            
            self.update_step = graph.get_operation_by_name("update_step")
            
            self.sigmoid_output = tf.sigmoid( self.output )
        
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
    
    def pretrain_agent(self, pt_input, pt_target):
        self.sess.run(self.pt_step, feed_dict={self.input: pt_input, self.pt_target: pt_target})
        
    def pretrain_loss(self, pt_input, pt_target):
        return self.sess.run(self.pt_loss, feed_dict={self.input: pt_input, self.pt_target: pt_target})
        
    def calculate_output(self, input_):
        return self.sess.run(self.sigmoid_output, feed_dict={self.input: input_})
    
    def calculate_loss(self, input_, action, reward):
        action_selector = np.column_stack((np.arange(len(action)), np.array(action)))        
        return self.sess.run(self.loss, feed_dict={self.input: input_, self.action: action_selector, self.target:reward})
    
    def update_agent(self, input_, action, reward):
        action_selector = np.column_stack((np.arange(len(action)), np.array(action)))        
        self.sess.run(self.update_step, feed_dict={self.input: input_, self.action: action_selector, self.target: reward})
        
    def save_model(self, dirpath):
        saver = tf.train.Saver()
        
        save_path = saver.save(self.sess, dirpath + "/model.ckpt")
        print("Model saved to: %s" % save_path)
        
        #tf.saved_model.simple_save(self.sess,
        #                           dirpath,
        #                           inputs={'input':self.input, 'action':self.action},
        #                           outputs={'output':self.output, 'target':self.target})

    def close_session(self):
        tf.reset_default_graph()
        self.sess.close()

def _weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    
def _bias_variable(shape, name):
    return tf.Variable(tf.constant(value=0.01, shape=shape), name=name)

def _leaky_relu(x):
    return tf.nn.relu(x) + 0.1 * tf.nn.relu(-x)

def _setup_pre_training(output, output_dimension, optimizer):
    pretrain_target = tf.placeholder(tf.float32, [None, output_dimension], name='pt_target')

    pretrain_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=pretrain_target)

    #pretrain_step = tf.train.AdamOptimizer(1e-6).minimize(pretrain_loss)
    pretrain_step = optimizer.minimize(pretrain_loss)
    
    return (pretrain_step, pretrain_target, pretrain_loss)

def _setup_training(output, output_dimension, optimizer):
    target = tf.placeholder(tf.float32, [None], name='target')
    action = tf.placeholder(tf.int32, [None, 2], name='action')
        
    out = tf.gather_nd(output, action)

#    loss = (tf.sigmoid(out) - target) ** 2
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=target, name='loss')

    update_step = optimizer.minimize(loss, name='update_step')
    
    return (update_step, action, target, loss)
    
def _construct_network(input_dimension, output_dimension, layer_sizes):
    input_= tf.placeholder(tf.float32, [None, input_dimension], name='input')

    nrof_layers = len(layer_sizes)
    
    # List of hidden layer variables
    W = []
    b = []
    h = []
    
    if nrof_layers > 0:
      # Define the first hidden layer
      with tf.name_scope('Hidden_Layer_0'):
          l = layer_sizes[0]
          W.append(_weight_variable([input_dimension, l], 'W0'))
          b.append(_bias_variable([l], 'b0'))
          
          h.append(tf.nn.relu(tf.matmul(input_, W[0]) + b))#, 'h0'))
      
      # Define the subsequent hidden layers
      for i in range(1, nrof_layers):
          with tf.name_scope('Hidden_Layer_%d'%(i)):
              l = layer_sizes[i] 
              W.append(_weight_variable([layer_sizes[i-1], l], 'W%d'%(i)))
              b.append(_bias_variable([l], 'b%d'%(i)))
              
              h.append(tf.nn.relu(tf.matmul(h[i-1], W[i]) + b[i]))#, name='h%d'%(i)))
      
      # Define the output layer
      W.append(_weight_variable([layer_sizes[-1], output_dimension], 'W%d'%(nrof_layers)))
      b.append(_bias_variable([output_dimension], 'b%d'%(nrof_layers)))

      output_ = tf.add(tf.matmul(h[-1], W[-1]), b[-1], name='output')

    else:
          W.append(_weight_variable([input_dimension, output_dimension], 'W0'))
          b.append(_bias_variable([output_dimension], 'b0'))

          output_ = tf.add( tf.matmul(input_, W[-1]), b[-1], name='output' )
    
    return (input_, output_)  
