import sys
sys.path.append("/global/home/users/avirukt/name/")
from jia import *

tf.enable_eager_execution()
warnings.filterwarnings('ignore')

def input_fn(version,n=1000,batch_size=256):
    parse = parse_fn(version)
    def fn():
        files = ["/global/scratch/avirukt/jia_sims/%s/9%03d.tfrecord"%(version,i) for i in range(n)] #use last 1k tfrecords for testing
        return tf.data.TFRecordDataset(files).map(parse).batch(batch_size)
    
    return fn

lower = np.array([0, 0.15, 0.45])
upper = np.array([.65, 0.45, 1.1])
bounds = list(zip(lower,upper))

p = load("../params.npy")
a = p[2:16]
p[16] = p[1]
p[1:15] = a