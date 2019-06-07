from power_spectrum import *
import sys

model = LFI(["field"], 
            [r"$M_\nu$",r"$\Omega_m$",r"$A_s\times10^9$"], 
            model_dir=sys.argv[1], 
            learning_rate=lambda x: tf.train.exponential_decay(0.001, x, 20000, 0.3, staircase=True)
           )

model.train(
    training_fn_generator(load("data_peak_counts/x_train.npy"), load("data_peak_counts/p_train.npy"), batch_size=128), 
    max_steps=100000
)
