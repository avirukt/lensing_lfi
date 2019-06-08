from power_spectrum import *
import sys

batch_size = 64

feature_description = {
    'field': tf.FixedLenFeature([2**16], tf.float32),
    "params": tf.FixedLenFeature([3], tf.float32)
}

data_path = ["/global/scratch/avirukt/jia_sims/%s/%04d.tfrecord"%(sys.argv[2],i) for i in range(9000)]
buffer_size = int(1.2*batch_size*256**2*4)

def parse(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    d = tf.parse_single_example(example_proto, feature_description)
    return (tf.reshape(d["field"],(256,256)),d["params"])

def training_input_fn(shuffle_buffer=100, batch_size=batch_size):
    dataset = tf.data.TFRecordDataset(data_path, buffer_size=buffer_size)
    return dataset.map(parse).repeat().shuffle(shuffle_buffer).batch(batch_size)

model = LFI(["field"], [r"$M_\nu$",r"$\Omega_m$",r"$\sigma_8$"], model_dir=sys.argv[1])
model.train(training_input_fn, max_steps=20000)
