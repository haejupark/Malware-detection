import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
import ast
import tensorflow as tf
#import matplotlib.pyplot as plt


train_filename =""
test_filename = ""
fileout = ""

# na for benign, otherwise malware
familyname = ["na","AdWo","Boxer","Dowgin","AirPush","Gappusin","SMStado","Counterclank","Wapsx","OpFake","SMSAgent"]

#train data and label
X = []
Y = []
with open(train_filename) as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    fname = row['filename']
    sus_b_c = row['sus_b_c']
    permission = row['permission']
    label = row['label']
    x = ast.literal_eval(sus_b_c)
    p = ast.literal_eval(permission)
    l = ast.literal_eval(label)
    X.append(x+p)
    Y.append(l)

#test data
Z = []
#filenaem list
fnames = []
with open(fileout) as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    fname = row['filename']
    fnames.append(fname)
    sus_b_c = row['sus_b_c']
    permission = row['permission']
    x = ast.literal_eval(sus_b_c)
    p = ast.literal_eval(permission)
    Z.append(x+p)

#split data for test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.33)

#parameters
learning_rate = 0.03
training_epochs = 1000
batch_size = 128
display_step = 1

n_hidden_1 = 200
n_hidden_2 = 200
n_input = 101
n_classes = 11

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
reg = 0

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

for key in weights:
    reg += tf.nn.l2_loss(weights[key])

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
prediction = tf.argmax(pred,1)

with tf.name_scope("cost") as scope:
  # Define cost function and regularization
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
  cost += 0.001 * reg
  c_summary = tf.summary.scalar('cost',cost)

with tf.name_scope("optimizer") as scope:
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("test") as scope:
  #test model
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  #calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.summary.scalar("accuracy",accuracy)

merged = tf.summary.merge_all()
# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    max_acc = 0
    max_arr = []

    writer = tf.summary.FileWriter('./board/sample_1',sess.graph)
 
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(Y_train, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

            answer = prediction.eval(feed_dict={x:Z})
            #print(answer)

        # Display logs per epoch step
        if epoch % display_step == 0:
            #a_summary = tf.summary.scalar('acc',accuracy)
            acc = accuracy.eval({x: X_test, y: Y_test})
            if acc > max_acc:
              max_acc = acc
              max_arr = answer
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),"Accuracy:", acc, "Max accuracy:", max_acc)
          
            summary = sess.run([merged,accuracy], feed_dict={x: X_test, y: Y_test})
            writer.add_summary(summary[0],epoch)
    #print(max_arr)
    #print(len(max_arr))        
    #print(len(fnames))
    with open(fileout,"w") as csvfile:
      fieldnames = ['filename','class','family']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for i in range(len(max_arr)):
        fname = fnames[i]
        cla = max_arr[i]
        family = familyname[int(cla)]
        if cla > 0:
          cla = 1
        writer.writerow({'filename': fname,'class' : cla, 'family':family})  
