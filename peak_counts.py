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

save("data_peak_counts/x_train.npy", x_train)
save("data_peak_counts/x_test.npy", x_test)
save("data_peak_counts/p_train.npy", p_train)
save("data_peak_counts/p_test.npy", p_test)

batch_size = 128

model = LFI(["field"], labels, model_dir=sys.argv[1], learning_rate=lambda x: tf.train.exponential_decay(0.001, x, 30000, 0.1, staircase=True))

model.train(training_fn_generator(x_train,p_train, batch_size=batch_size), max_steps=100000)
