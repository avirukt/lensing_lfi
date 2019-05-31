from power_spectrum import *
import sys

nsims_test = 10**4
batch_size = 128

feature_description = {
    'field': tf.FixedLenFeature([2**16], tf.float32),
    "params": tf.FixedLenFeature([3], tf.float32)
}

data_path = "/global/scratch/avirukt/jia_sims.tfrecord"
buffer_size = int(1.2*batch_size*256**2*4)

def parse(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    d = tf.parse_single_example(example_proto, feature_description)
    return (tf.reshape(d["field"],(256,256)),d["params"])

def testing_input_fn(batch_size=batch_size):
    dataset = tf.data.TFRecordDataset(data_path, buffer_size=buffer_size)
    return dataset.take(nsims_test).map(parse).batch(batch_size)

def training_input_fn(shuffle_buffer=100, batch_size=batch_size):
    dataset = tf.data.TFRecordDataset(data_path, buffer_size=buffer_size)
    return dataset.skip(nsims_test).map(parse).repeat().shuffle(shuffle_buffer).batch(batch_size)


model = LFI(["field"], [r"$M_\nu$",r"$\Omega_m$",r"$\sigma_8$"], model_dir=sys.argv[1])
model.train(training_input_fn, max_steps=3*10**6//batch_size)
