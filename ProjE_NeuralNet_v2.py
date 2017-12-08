import argparse
import math
import os.path
import timeit
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf



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
        f_predpath[i][j] = int(f_predpath[i][j])
train_predpath = np.array(f_predpath)


f_triple = open('./data_id/city_capital.csv','r')
f_triple  = f_triple.readlines()
f_triple = f_triple[1:]
f_triple = [f_triple[i].rstrip('\n').split(',')[:-1] for i in range(len(f_triple))]
for i in range(len(f_triple)):
    for j in range(len(f_triple[0])):
        f_triple[i][j] = int(f_triple[i][j])
f_triple = np.array(f_triple)
hrt_triples = f_triple

n_predicate = train_predpath.shape[1]-1
print("N_TRAIN_TRIPLES: %d" % train_predpath.shape[0])

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 100 # 1st layer number of neurons
n_hidden_2 = 1 # 2nd layer number of neurons
n_input =  n_predicate# MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

embed_dim = n_hidden_1
bound = 6 / math.sqrt(n_hidden_1)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)
    
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()



with tf.Session() as session:
    tf.global_variables_initializer().run()

    kf = KFold(n_splits=10, random_state=1233)
    print("Initializing 10-folds training data...")
    i_fold = 1
    for i_train, i_test in kf.split(train_predpath):
        train_predicates = np.array(train_predpath)[i_train]
        test_predicates = np.array(train_predpath)[i_test]
        train_tiples = np.array(hrt_triples)[i_train]    
        
        for epoch in range(training_epochs):

            accu_loss = 0.
            ninst = 0                 
            #print("Minibatches training... iteration: ", n_iter)           

            X_batch = train_predicates[:,1:]
            tmp = train_predicates[:,0]
            Y_labels = np.zeros([X_batch.shape[0], n_classes])
            for j in range(X_batch.shape[0]):
                Y_labels[j,tmp[j]] = 1.
            _, c = session.run([train_op, loss_op], feed_dict = 
                               {X: X_batch, Y: Y_labels})
                #accu_loss += l
            #print("Loss ", accu_loss)
        print("Finish training data. Fold ", i_fold)       
        i_fold = i_fold + 1
        
        print("Evaluation")
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        tmp_ = test_predicates[:,0]
        test_labels = np.zeros([len(tmp_), n_classes])
        for j in range(len(tmp_)):
            test_labels[j,tmp_[j]] = 1.
        print("Accuracy:", accuracy.eval({X: test_predicates[:,1:], Y: test_labels}))
        a = tf.cast(tf.argmax(pred, 1),tf.float32)
        b = tf.cast(tf.argmax(Y,1),tf.float32)
        
        auc = tf.contrib.metrics.streaming_auc(a, b)
        session.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
        train_auc = session.run(auc, feed_dict={X: test_predicates[:,1:], Y: test_labels})
        
        print(train_auc)
#            if (pred_prob>0.5):
#                pred_label = 1
#            else:
#                pred_label = 0
        
        #pred_label = np.argmax(pred_prob, axis=1)
        #return np.vstack([1 - pred_prob, pred_prob]).T        
    
