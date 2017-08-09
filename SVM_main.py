import sys
from non_shrinkingSVM_parallel import *
from non_shrinkingSVM_singlenode import *
from shrinkedSVM_parallel import fit_shrinking_SVM_parallel


def main(argv):

    mode = ''
    fit = ''
    test = ''
    cls = ''
    workers = 1
    samples = 1000
    master = 'grpc://localhost:2222'
    shrink = ''
    shrinking_counter = 50
    type='mnist'

    if '-p' in argv:
        index = argv.index('-p')
        mode = argv[index]
    if '-w' in argv:
        index = argv.index('-w')
        workers = int(argv[index + 1])
    if '-f' in argv:
        index = argv.index('-f')
        fit = argv[index]
    if '-t' in argv:
        index = argv.index('-t')
        test = argv[index]
    if '-c' in argv:
        index = argv.index('-c')
        cls = int(argv[index + 1])
    if '-s' in argv:
        index = argv.index('-s')
        samples = int(argv[index + 1])
    if '-m' in argv:
        index = argv.index('-m')
        master = argv[index + 1]
    if '-sh' in argv:
        index = argv.index('-sh')
        shrink = argv[index]
    if '-th' in argv:
        index = argv.index('-th')
        shrinking_counter = int(argv[index + 1])
    if '-type' in argv:
        index = argv.index('-type')
        type = argv[index + 1]

    if mode == '-p':
        # call parallel fit
        # start master session on any of the workers
        with tf.Session(master) as sess:
            if fit == '-f':
                if shrink == '-sh':
                    b = fit_shrinking_SVM_parallel(sess, cls, workers, samples, shrinking_counter, type)
                else:
                    b = fit_SVM_parallel(sess, cls, workers, samples, type)
            if test == '-t':
                test_SVM_parallel(sess, b, cls, workers, type)
    else:
        with tf.Session() as sess:
            if fit == '-f':
                b = fit_SVM_single_worker(sess, cls, samples, type)
            if test == '-t':
                test_SVM_single_worker(sess, b, cls, type)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])