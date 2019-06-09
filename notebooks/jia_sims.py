import sys
sys.path.append("/global/home/users/avirukt/name/")
from power_spectrum import *

tf.enable_eager_execution()
warnings.filterwarnings('ignore')

batch_size = 256

def input_fn(version,n=1000):
    feature_description = {
        'field': tf.FixedLenFeature([2**16], tf.float32),
        "params": tf.FixedLenFeature([3], tf.float32)
    }

    if "random_mask" in version:
        feature_description["mask"] = tf.FixedLenFeature([2**16], tf.float32)

        def parse(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            d = tf.parse_single_example(example_proto, feature_description)
            return (tf.reshape(tf.stack([d["field"],d["mask"]],axis=-1),(256,256,2)),d["params"])
    else:
        def parse(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            d = tf.parse_single_example(example_proto, feature_description)
            return (tf.reshape(d["field"],(256,256)),d["params"])

    def fn():
        files = ["/global/scratch/avirukt/jia_sims/%s/9%03d.tfrecord"%(version,i) for i in range(n)] #use last 1k tfrecords for testing
        return tf.data.TFRecordDataset(files).map(parse).batch(batch_size)
    
    return fn

lower = np.array([0, 0.15, 0.45])
upper = np.array([.65, 0.45, 1.1])
bounds = list(zip(lower,upper))

p = load("../params.npy")
a = p[2:16]
p[16] = p[1]
p[1:15] = a