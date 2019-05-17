# lensing_lfi

## Jia's processed sims

- `jia_sims.tfrecord`: ~250 GB (101x10000 maps), 256x256 pixels, 0.8 arcmin each
- `jia_sims_noisy.tfrecord`: ~66 GB, including noise (standard deviation of 0.082 in each pixel)
- `jia_sims_whitened.tfrecord`: ~66 GB, standardised to flat power spectrum (using interpolation)
- `jia_sims_gaussian.tfrecord`: ~66 GB, Gaussian sims with removed (interpolated) power spectrum

Beware the last few might be corrupted (trust upto the penultimate batch, batch size was 1k). Every 101 elements are the 101 cosmologies.

The files are in:
- NERSC: `/global/cscratch1/sd/avirukt/jia_sims/`
- Savio: `global/scratch/avirukt/`

Each element is dict with two keys: `field`, a list of 256x256 floats, and `params`, a list of 3 floats (M_nu, Omega_m, sigma_8). To generate testing and training input functions for `tf.Estimator`s:

```
feature_description = {
    'field': tf.FixedLenFeature([2**16], tf.float32),
    "params": tf.FixedLenFeature([3], tf.float32)
}

def parse(example_proto):
    d = tf.parse_single_example(example_proto, feature_description)
    return (tf.reshape(d["field"],(256,256)),d["params"])

def testing_input_fn():
    dataset = tf.data.TFRecordDataset(tfrecord_path, buffer_size=2**30)
    return dataset.take(nsims_test).map(parse).batch(batch_size)

def training_input_fn():
    dataset = tf.data.TFRecordDataset(tfrecord_path, buffer_size=2**30)
    return dataset.skip(nsims_test).map(parse).repeat().shuffle(shuffle_buffer).batch(batch_size)
```
