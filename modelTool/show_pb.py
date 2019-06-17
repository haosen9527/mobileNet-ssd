import tensorflow as tf
with tf.Session() as sess:
    with open(sys.argv[1], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        print graph_def
