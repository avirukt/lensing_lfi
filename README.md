# lensing_lfi

## Jia's processed sims

- `jia_sims.tfrecord`: ~250 GB (101x10000 maps), 256x256 pixels, 0.8 arcmin each
- `jia_sims_noisy.tfrecord`: ~66 GB, including noise (standard deviation of 0.082 in each pixel)
- `jia_sims_whitened.tfrecord`: ~66 GB, standardised to flat power spectrum (using interpolation)
- `jia_sims_gaussian.tfrecord`: ~66 GB, Gaussian sims with removed (interpolated) power spectrum

Beware the last few might be corrupted (trust upto the penultimate batch, batch size was 1k)

The files are in:
- NERSC: `/global/cscratch1/sd/avirukt/jia_sims/`
- Savio: `global/scratch/avirukt/`
