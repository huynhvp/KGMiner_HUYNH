import argparse
import math
import os.path
import timeit
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf

class ProjE:
    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def pred_embedding(self):
        return self.__ent_embedding

    @property
    def training_predicate(self):
        return self.__train_predpath
    
    @property
    def hrt_tiples(self):
        return self.__hrt_triples

    def __init__(self, embed_dim=100):

        self.__embed_dim = embed_dim
        self.__initialized = False

        self.__trainable = list()
        #self.__dropout = dropout

        f_predpath = open('./Predicate_paths/city_capitals_1.csv','r')
        f_predpath = f_predpath.readlines()
        f_predpath = f_predpath[1:]
        f_predpath = [f_predpath[i].rstrip('\n').split(',')[1:] for i in range(len(f_predpath))]
        for i in range(len(f_predpath)):
            for j in range(len(f_predpath[0])):
                if (f_predpath[i][j] == 'TRUE'): 
                    f_predpath[i][j] = 1
                if (f_predpath[i][j] == 'FALSE'):
                    f_predpath[i][j] = 0
                f_predpath[i][j] = float(f_predpath[i][j])
        f_predpath = np.array(f_predpath)
        self.__train_predpath = f_predpath


        f_triple = open('./data_id/city_capital.csv','r')
        f_triple  = f_triple.readlines()
        f_triple = f_triple[1:]
        f_triple = [f_triple[i].rstrip('\n').split(',')[:-1] for i in range(len(f_triple))]
        for i in range(len(f_triple)):
            for j in range(len(f_triple[0])):
                f_triple[i][j] = float(f_triple[i][j])
        f_triple = np.array(f_triple)
        self.__hrt_triples = f_triple
        
        self.n_predicate = self.__train_predpath.shape[1]-1
        print("N_TRAIN_TRIPLES: %d" % self.__train_predpath.shape[0])
        
        bound = 6 / math.sqrt(embed_dim)

        with tf.device('/cpu'):
            self.__pred_embedding = tf.get_variable("pred_embedding", [self.n_predicate, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=345))
            self.__trainable.append(self.__pred_embedding)   
            self.__pred_bias = tf.get_variable("pred_bias",[embed_dim,1],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=345))
            self.__trainable.append(self.__pred_bias)

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor , 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp, 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * 1  # all ignored elements will have a prob of 0.
    
    def train(self, inputs, regularizer_weight=0., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()

            pred_list, pred_weight = inputs


            pred_res = tf.matmul(tf.matmul(pred_list, self.__pred_embedding), self.__pred_bias)
                                      
            #pred_res_sigmoid = tf.sigmoid(pred_res)
            #self.pred_softmax = pred_res_softmax = self.sampled_softmax(pred_res, pred_weight)
            #self.pred_softmax = pred_res_softmax 
            #pred_loss = -tf.reduce_sum(
            #    tf.log(tf.clip_by_value(pred_res_softmax, 1e-10, 1.0)) * pred_weight)
           #     + tf.log(tf.clip_by_value(1 - pred_res_sigmoid, 1e-10, 1.0)) * (1-pred_weight))
            pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_res, labels=pred_weight)
            self.regularizer_loss = regularizer_loss = tf.reduce_sum(
                tf.abs(self.__pred_embedding)) 
          
            return pred_loss + regularizer_loss * regularizer_weight

    def test(self, inputs, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()
            pred_list = inputs
            pred_prob = tf.matmul(tf.matmul(pred_list, self.__pred_embedding), self.__pred_bias)
#            print type(pred_prob)
#            pred_prob *= -1
#            pred_prob = tf.exp(pred_prob)
#            #np.exp(pred_prob, pred_prob)
#            pred_prob += 1
#            #np.reciprocal(pred_prob, pred_prob)
#            pred_prob = tf.reciprocal(pred_prob)
            
#            if (pred_prob>0.5):
#                pred_label = 1
#            else:
#                pred_label = 0
            
            #pred_label = np.argmax(pred_prob, axis=1)
            #return np.vstack([1 - pred_prob, pred_prob]).T
            return pred_prob

def train_ops(model, learning_rate=0.01, optimizer_str='adam', regularizer_weight=0.0):
    with tf.device('/cpu'):
        pred_input = tf.placeholder(tf.float32, [None, model.n_predicate])
        pred_weight = tf.placeholder(tf.float32, [None,1])

        loss = model.train([pred_input, pred_weight],
                           regularizer_weight=regularizer_weight)
        if optimizer_str == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError("Does not support %s optimizer" % optimizer_str)

        grads = optimizer.compute_gradients(loss, model.trainable_variables)

        op_train = optimizer.apply_gradients(grads)

        return pred_input, pred_weight, loss, op_train
    
def test_ops(model):
    with tf.device('/cpu'):
        test_input = tf.placeholder(tf.float32, [None, model.n_predicate])
        pred_probs = model.test(test_input)

    return test_input, pred_probs
    
def main(_):
    parser = argparse.ArgumentParser(description='ProjE.')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=100)
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=100)
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",
                        default=0.5)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
    parser.add_argument("--loss_weight", dest='loss_weight', type=float, help="Weight on parameter loss", default=0.0)



    args = parser.parse_args()

    print(args)

    model = ProjE(embed_dim=args.dim)
    
    pred_input, pred_weight, train_loss, train_op = train_ops(model, learning_rate=args.lr, optimizer_str=args.optimizer, regularizer_weight=args.loss_weight)
    test_input, test_pred_prob =  test_ops(model)
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        kf = KFold(n_splits=10)
        print("Initializing 10-folds training data...")
        i_fold = 1
        for i_train, i_test in kf.split(model.training_predicate):
            train_predicates = np.array(model.training_predicate)[i_train]
            test_predicates = np.array(model.training_predicate)[i_test]
            train_tiples = np.array(model.hrt_tiples)[i_train]    
            
            for n_iter in range(args.max_iter):
                start_time = timeit.default_timer()
                accu_loss = 0.
                accu_re_loss = 0.
                ninst = 0                 
                #print("Minibatches training... iteration: ", n_iter)
                
                head_unique = np.unique(train_tiples[:,0])
                for i_head in head_unique:
                    l, _ = session.run([train_loss, train_op], 
                                       {pred_input: train_predicates[train_tiples[:,0]==i_head,1:], pred_weight: train_predicates[train_tiples[:,0]==i_head,0][np.newaxis].T})
                    #accu_loss += l
                #print("Loss ", accu_loss)
            print("Finish training data. Fold ", i_fold)       
            i_fold = i_fold + 1
            
            print("Evaluation")
            predict_proba= session.run([test_pred_prob],
                                                       {test_input: test_predicates[:,1:]})
            print predict_proba[0]
            predict_proba = predict_proba[0]
            predict_proba *= -1
            np.exp(predict_proba, predict_proba)
            predict_proba += 1
            np.reciprocal(predict_proba, predict_proba)
            predict_label = np.copy(predict_proba);
            predict_label[predict_label>0.5] = 1
            predict_label[predict_label<0.5] = 0
            print predict_label
            print test_predicates[:,0]
#            if (pred_prob>0.5):
#                pred_label = 1
#            else:
#                pred_label = 0
            
            #pred_label = np.argmax(pred_prob, axis=1)
            #return np.vstack([1 - pred_prob, pred_prob]).T        
        
        
        
if __name__ == '__main__':
    tf.app.run()
    