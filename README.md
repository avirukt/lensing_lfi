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


## Running on Savio

The file `jia.py` is the script for Jia's sims, `peak_counts.py` are for the peak count histograms. 

First, make sure to commit any changes since the run will be identified with the (short) commit hash. Make sure there are folders named `outputs` and `scripts` in the repo root directory. Run the file `run_script.py` passing the name of the script as an argument. The script is expected to take one argument, which is the name of the model directory. The name of the run is the name of the script with the commit hash appended to it. This is the structure for `peak_counts.py`, e.g. you would run `python run_script.py peak_counts.py` and that would create a job `scripts/peak_sounts-4dc0c6b.sh` and submit it. The job will create a model directory `/global/scratch/avirukt/models/peak_counts-4dc0c6b` and output `outputs/peak_counts-4dc0c6b.out`. 

`jia.py` works differently since it takes two arguments: the model directory, and the name of the version (to get the right data and architecture). `run_script.py` uses the version name instead of the script name for naming the job/model_dir/etc. and so you must supply the version as well e.g. `python run_script.py jia.py noiseless` will create a job with name/script/output/model_dir `noiseless-4dc0c6b`. 

You will need Python 3 and GitPython in the environment from which you submit the job, and you will need to create a virtual/conda env named `tensorflow` with tfp 0.5 installed to actually run the job.