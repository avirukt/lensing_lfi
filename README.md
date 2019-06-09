# lensing_lfi

## Jia's processed sims

There are 101x10k sims, which are in 10k TFRecords named `0000.tfrecord` to `9999.tfrecord`. 

There are 4x4=16 versions: 
4 kinds of pre-processing:

1. No pre-processing
2. Whitened
3. Partially whitened
4. Gaussian (with removed PS)

4 kinds of noise:

1. Noiseless
2. Constant noise: use average of 5.12 galaxies for each pixel
3. Constant "mask": draw a single galaxy map from Poisson and use it for all images
4. Random "mask": draw a random galaxy map for each image

Each version is a folder in:
- NERSC: `/global/cscratch1/sd/avirukt/jia_sims/`
- Savio: `global/scratch/avirukt/jia_sims/`

The name of the folder containing the TFRecords for each version is 

||1|2|3|4|
|---|---|---|---|---|
|1|`noiseless`|`noiseless_whitened`||`noiseless_gaussian`|
|2|`constant_noise`||||
|3|`constant_mask`\*||||
|4|`random_mask`||||

\*The directory `constant_mask` also contains the file `mask.npy` which contains the galaxy map drawn from a Poisson that was used for the maps in that folder.

Each TFRecord contains 101 elements corresponding to the 101 cosmologies. Each element is dict with two keys: `"field"`, a list of 256x256 floats, and `"params"`, a list of 3 floats (M_nu, Omega_m, sigma_8). The maps with random "masks" also contains the key `"mask"` a list of 256x256 32-bit integers (too late to change to bytes). Thus each TFRecord is 26 MB (51 MB with "mask"). To generate testing and training input functions for `tf.Estimator`s:

```
feature_description = {
    'field': tf.FixedLenFeature([2**16], tf.float32),
    "params": tf.FixedLenFeature([3], tf.float32)
}

def parse(example_proto):
    d = tf.parse_single_example(example_proto, feature_description)
    return (tf.reshape(d["field"],(256,256)),d["params"])

def input_fn():
    dataset = tf.data.TFRecordDataset(tfrecord_paths, buffer_size=buffer_size)
    return dataset.map(parse)
```
