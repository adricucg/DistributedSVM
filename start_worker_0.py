from util import *

tf.logging.set_verbosity(tf.logging.DEBUG)

# start_worker_server('job1',
#                      ["172.31.35.116:2222", "172.31.23.179:2222", "172.31.29.90:2222", "172.31.24.80:2222",
#                       "172.31.17.227:2222", "172.31.18.189:2222", "172.31.25.51:2222", "172.31.27.140:2222"],
#                      0)

start_worker_server('job1',
                    ["localhost:2222", "localhost:2223", "localhost:2224", "localhost:2225", "localhost:2226"],
                    0)

start_worker_server('job1',
                    ["172.31.24.12:2222", "172.31.23.220:2222", "172.31.16.76:2222", "172.31.28.213:2222",
                     "172.31.22.65:2222", "172.31.30.170:2222", "172.31.16.199:2222", "172.31.24.204:2222",
                     "172.31.24.101:2222", "172.31.19.128:2222", "172.31.24.183:2222", "172.31.16.8:2222",
                     "172.31.17.76:2222", "172.31.31.86:2222", "172.31.19.132:2222", "172.31.27.150:2222"
                    ],
                     1)