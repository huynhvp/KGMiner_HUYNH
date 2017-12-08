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
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 200 # 1st layer number of neurons
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

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

with tf.Session() as session:
    tf.global_variables_initializer().run()
    kf = KFold(n_splits=10)
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
            head_unique = np.unique(train_tiples[:,0])
            for i_head in head_unique:
                _, c = session.run([train_op, loss_op], feed_dict = 
                                   {X: train_predicates[train_tiples[:,0]==i_head,1:], Y: train_predicates[train_tiples[:,0]==i_head,0]})
                #accu_loss += l
            #print("Loss ", accu_loss)
        print("Finish training data. Fold ", i_fold)       
        i_fold = i_fold + 1
        
        print("Evaluation")
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: test_predicates[:,1:], Y: test_predicates[:,0]}))
#            if (pred_prob>0.5):
#                pred_label = 1
#            else:
#                pred_label = 0
        
        #pred_label = np.argmax(pred_prob, axis=1)
        #return np.vstack([1 - pred_prob, pred_prob]).T        
    
