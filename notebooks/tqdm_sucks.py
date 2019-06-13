import sys
import time
from dateutil.relativedelta import relativedelta

attrs = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
human_readable = lambda delta: ['%d %s' % (getattr(delta, attr), getattr(delta, attr) > 1 and attr or attr[:-1]) for attr in attrs if getattr(delta, attr)]
showtime = lambda x: " ".join(human_readable(relativedelta(seconds=x))[:2])

def tqdm(iterator,maxsize,step=1):
    start = time.time()
    cp = 0
    for i,x in enumerate(iterator):
        now = time.time()-start
        if now-cp > step and i>0:
            cp = now
            sys.stdout.flush()
            sys.stdout.write("\r%d/%d~%.1f%%, %s so far, %s longer"%(i+1,maxsize,100*i/maxsize,showtime(now),showtime(now*(maxsize/i-1))))
        yield x