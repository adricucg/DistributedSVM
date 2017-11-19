from util import *

tf.logging.set_verbosity(tf.logging.DEBUG)

start_worker_server('job1',
                    ["localhost:2222", "localhost:2223", "localhost:2224", "localhost:2225", "localhost:2226"],
                    0)