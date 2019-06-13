from power_spectrum import *
import sys

batch_size = 64
version = sys.argv[2]

feature_description = {
    'field': tf.FixedLenFeature([2**16], tf.float32),
    "params": tf.FixedLenFeature([3], tf.float32)
}

scale = {
    "noiseless": 0.16,
    "noiseless_gaussian": 0.147,
    "noiseless_whitened": 1.06,
}[version]

cutoff = 5
    
def parse(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    d = tf.parse_single_example(example_proto, feature_description)
    field = tf.clip_by_value(d["field"]/scale,-cutoff,cutoff)
    return (tf.reshape(field,(256,256)),d["params"])

def training_input_fn(shuffle_buffer=100, batch_size=batch_size):
    files = tf.data.Dataset.list_files("/global/scratch/avirukt/jia_sims/%s/[0-8]*.tfrecord"%version) #save last 1k tfrecords for testing
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=2, block_length=1, num_parallel_calls=2)
    return dataset.map(parse).repeat().shuffle(shuffle_buffer).batch(batch_size)

model = LFI(
    ["field"], 
    [r"$M_\nu$",r"$\Omega_m$",r"$\sigma_8$"], 
    model_dir=sys.argv[1], 
    learning_rate = lambda x: tf.train.exponential_decay(0.0005, x, 2000, 0.3, staircase=False)
)

model.train(training_input_fn, max_steps=20000)