import tensorflow as tf


print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.bool, shape=[])
    
    def f1(): return tf.constant(1)
    def f2(): return tf.constant(2)
    r = tf.cond(x, f1, f2)

with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()

    res = session.run([r],feed_dict={x : True})
    print(res)
