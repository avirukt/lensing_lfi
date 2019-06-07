from power_spectrum import *
import sys

p = np.load("params_conc.npy")
x = np.load("data_scaled.npy")
tot_nsims = len(p)

labels = [r"$M_\nu$",r"$\Omega_m$",r"$A_s\times10^9$"]
nsims_train = 10**4
nsims_test = tot_nsims - nsims_train

indices = np.arange(tot_nsims)
random.shuffle(indices)
x_train = array([x[ind] for ind in indices[:nsims_train]])
p_train = array([p[ind] for ind in indices[:nsims_train]])
x_test = array([x[ind] for ind in indices[nsims_train:]])
p_test = array([p[ind] for ind in indices[nsims_train:]])

batch_size = 128

model = LFI(["field"], labels, model_dir=sys.argv[1], learning_rate=lambda x: tf.cond(x<5000, lambda: 0.001, lambda: 0.0001))

model.train(training_fn_generator(x_train,p_train, batch_size=batch_size), max_steps=100000)
