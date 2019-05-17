from power_spectrum import *

p = np.load("params_conc.npy")
x = np.load("data_scaled.npy")
tot_nsims = len(p)

labels = [r"$M_\nu$",r"$\Omega_m$",r"$A_s\times10^9$"]
nsims_train = 10**4
nsims_test = tot_nsims - nsims_train

indices = np.arange(tot_nsims)
random.shuffle(indices)
x_train = array([small_data[ind] for ind in indices[:nsims_train]])
p_train = array([params[ind] for ind in indices[:nsims_train]])
x_test = array([small_data[ind] for ind in indices[nsims_train:]])
p_test = array([params[ind] for ind in indices[nsims_train:]])

batch_size = 128

model = LFI(["field"],labels,model_dir="/global/scratch/avirukt/models/peak_counts")

model.train(training_fn_generator(x_train,p_train, batch_size=batch_size), max_steps=30*nsims_train//batch_size)
