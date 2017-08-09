import tensorflow as tf

sess = tf.InteractiveSession()

gradient_new = tf.placeholder(tf.float32)
g_new = [[1,12],[1,12]]

p = True

with tf.variable_scope("svm_worker") as scope:

    gradient_old = tf.get_variable('gradient_old', initializer=tf.zeros(shape=(2,2)))

    eliminated_samples = tf.get_variable('wq', initializer=tf.zeros(shape=gradient_old.shape))
    term = tf.add(gradient_old, gradient_new)
    ass = tf.assign(gradient_old, term)

    min_g = tf.reduce_min(ass, 1)
    max_g = tf.reduce_max(ass, 1)

    scope.reuse_variables()

    sess.run(tf.global_variables_initializer())

    res1 = sess.run([min_g, max_g], feed_dict={gradient_new: g_new})
    res2 = sess.run([min_g, max_g], feed_dict={gradient_new: g_new})

    print(res1)
    print(res2)
    print(sess.run(eliminated_samples))
    #sess.run(gradient_reshaped)

    if p:
        const = tf.constant(-198)

        #sess.run(tf.local_variables_initializer())

        print(sess.run(const))
        print(sess.run(gradient_old))




    #print(sess.run(min_g, feed_dict={gradient_new: g_new}))
    #print(sess.run(max_g, feed_dict={gradient_new: g_new}))

