from numpy import *
from numpy.random import rand,randn,randint
from itertools import permutations
from scipy.stats import binned_statistic,binom
from matplotlib.pyplot import *
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.interpolate import LSQUnivariateSpline

colon=slice(None)

def field_from_spectrum(ps,boxsize=128,n=1):
    f=randn(n,boxsize,boxsize)*exp(2j*pi*rand(n,boxsize,boxsize))
    for i in range(boxsize//2+1):
        for j in range(boxsize):
            f[:,i,j]=f[:,i,j]*ps((i**2+min(j,boxsize-j)**2)**.5)**.5
    f[:,0,0]=0
    if not boxsize%2:
        f[:,0,boxsize//2]=abs(f[:,0,boxsize//2])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,0]=abs(f[:,boxsize//2,0])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,boxsize//2]=abs(f[:,boxsize//2,boxsize//2])*(randint(2,size=n)*2-1)
        for i in range(boxsize//2+1,boxsize):
            f[:,boxsize//2,i]=conj(f[:,boxsize//2,boxsize-i])
    for j in range(boxsize//2+1,boxsize):
        f[:,0,j]=conj(f[:,0,boxsize-j])
        f[:,j,0]=conj(f[:,boxsize-j,0])
        for i in range(1,boxsize):
            f[:,j,i]=conj(f[:,boxsize-j,boxsize-i])
    return real(fft.fft2(f))

def grf_from_spectrum(ps,boxsize=128,n=1):
    f=(randn(n,boxsize,boxsize)+1j*randn(n,boxsize,boxsize))/2
    for i in range(boxsize//2+1):
        for j in range(boxsize):
            f[:,i,j]=f[:,i,j]*ps((i**2+min(j,boxsize-j)**2)**.5)**.5
    f[:,0,0]=0
    if not boxsize%2:
        f[:,0,boxsize//2]=abs(f[:,0,boxsize//2])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,0]=abs(f[:,boxsize//2,0])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,boxsize//2]=abs(f[:,boxsize//2,boxsize//2])*(randint(2,size=n)*2-1)
        for i in range(boxsize//2+1,boxsize):
            f[:,boxsize//2,i]=conj(f[:,boxsize//2,boxsize-i])
    for j in range(boxsize//2+1,boxsize):
        f[:,0,j]=conj(f[:,0,boxsize-j])
        f[:,j,0]=conj(f[:,boxsize-j,0])
        for i in range(1,boxsize):
            f[:,j,i]=conj(f[:,boxsize-j,boxsize-i])
    return real(fft.fft2(f))

def simple_grf(ps,boxsize=128,n=1):
    f=(randn(n,boxsize,boxsize)+1j*randn(n,boxsize,boxsize))/2**.5
    for i in range(boxsize):
        for j in range(boxsize):
            f[:,i,j]=f[:,i,j]*ps((min(i,boxsize-i)**2+min(j,boxsize-j)**2)**.5)**.5
    f[:,0,0]=0
    return real(fft.fft2(f))

def build_ps(arr, ps, size, d, indices=[]):
    if len(indices)==d:
        psvals = ps(sum(array(indices,dtype=float)**2)**.5)
        for index in permutations(indices):
            for i in range(2**(d-1)):
                ind = list(index)
                for k in range(d-1):
                    if i//2**k % 2 and ind[k]>0:
                        ind[k] = size-ind[k]
                arr[tuple([colon]+ind)] = psvals
        return
    if indices==[]:
        [build_ps(arr,ps,size,d,[i]) for i in range(size//2+1)]
        return
    [build_ps(arr,ps,size,d,indices+[i]) for i in range(indices[-1]+1)]

def fast_field(ps, size=32, d=2, fnl=0, fnl_potential=True):
    n = len(ps(0))
    assert not size%2
    ls = size//2+1
    shape=[n]+[size]*(d-1)+[ls]
    field = randn(*shape)*exp(2j*pi*rand(*shape))
    constants = [0, size//2]
    for i in range(2**d):
        specials = tuple([colon]+[constants[(i//2**k)%2] for k in range(d)])
        field[specials] = abs(field[specials])*(randint(2,size=n)*2-1)
    k2 = zeros(shape)
    build_ps(k2,ps,size,d)
    field *= k2**.5
    field[tuple([colon]+[0]*d)] = 0
    factor = size**(d/2)
    ft = lambda x: fft.rfftn(x,axes=tuple(range(1,d+1)))/factor
    ift = lambda x: fft.irfftn(x,axes=tuple(range(1,d+1)))*factor
    if fnl==0:
        field = ift(field)
    elif fnl_potential:
        k2=zeros([1]+[size]*(d-1)+[ls])
        k2[tuple(np.zeros(d+1))]=1
        build_ps(k2,lambda x: x**2*ones(1),size,d)
        field /= k2
        field[tuple([colon]+[0]*d)] = 0
        field = ift(field)
        #field /= std(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)
        field += fnl*field**2
        field = ft(field)
        field[tuple([colon]+[0]*d)] = 0
        field = fft.irfftn(field*k2,axes=tuple(range(1,d+1)))
    else:
        field = fft.irfftn(field, axes=tuple(range(1,d+1)))
        #field /= std(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)
        field += fnl*field**2
        field -= mean(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)
    return field/std(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)

def sdft(x,axes=None):
    d = x.ndim-1
    size = x.shape[-1]
    n = x.shape[0]
    if axes is None:
        axes=tuple(range(1,d+1))
    factor = size**(d/2)
    return fft.fftn(x,axes=axes)/factor

def iterable(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False

def plot_2d_field(x,cbar=True):
    m = amax(abs(x))
    imshow(x, vmin=-m, vmax=m, cmap="bwr")
    if cbar:
        colorbar()

def fast_gaussian(ps, size=32, d=2, fnl=0, fnl_potential=True, constant_phase=False):
    n = len(ps(0))
    assert not size%2
    ls = size//2+1
    shape=[n]+[size]*(d-1)+[ls]
    field = (randn(*shape) + 1j*randn(*shape))/2**.5
    for i in range(2**d):
        specials = tuple([colon]+[((i//2**k)%2)*(size//2) for k in range(d)])
        field[specials] = randn(n)
    if constant_phase:
        field[:] = field[0].reshape([1]+shape[1:])
    k2 = zeros(shape)
    build_ps(k2,ps,size,d)
    field *= k2**.5
    field[tuple([colon]+[0]*d)] = 0
    axes=tuple(range(1,d+1))
    factor = size**(d/2)
    ft = lambda x: fft.rfftn(x,axes=axes)/factor
    ift = lambda x: fft.irfftn(x,axes=axes)*factor
    if iterable(fnl):
        fnl = array(fnl).reshape([n]+[1]*d)
    if not iterable(fnl) and fnl==0:
        field = ift(field)
    elif fnl_potential:
        k2=zeros([1]+[size]*(d-1)+[ls])
        build_ps(k2,lambda x: x**2*ones(1),size,d)
        field /= k2
        field[tuple([colon]+[0]*d)] = 0
        field = ift(field)
        field += fnl*field**2
        field = ft(field)
        field = ift(field*k2)
    else:
        field = ift(field)
        field += fnl*field**2
        field -= mean(field, axis=axes).reshape([n]+[1]*d)
    return field

def power_spectrum(x):
    size=x.shape[-1]
    kx,ky=arange(size),arange(size)
    kx,ky=minimum(kx,size-kx),minimum(kx,size-kx)
    kx,ky=meshgrid(kx,ky)
    k=sqrt(kx**2+ky**2).flatten()
    p=(abs(fft.ifft2(x))**2).flatten()
    avg,edges,f= binned_statistic(k,p)
    err,edges,f= binned_statistic(k,p**2)
    count,edges,f=binned_statistic(k,p,statistic="count")
    return (edges[:-1]+edges[1:])/2,avg,sqrt((err-avg**2)/count),count

def get_flt_k_f(x):
    size=x.shape[-1]
    kx,ky=arange(size),arange(size)
    kx,ky=minimum(kx,size-kx),minimum(kx,size-kx)
    kx,ky=meshgrid(kx,ky)
    k=sqrt(kx**2+ky**2).flatten()
    fk=fft.ifft2(x).flatten()
    return k,fk

def norm_10_bins(x):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    avg,edges,binnum= binned_statistic(k,p)
    #print(avg,edges,k[:10],binnum[:10])
    for i in range(size**2):
        fk[i]=fk[i]/avg[binnum[i]-1]**.5
    fk=fk.reshape((size,size))
    return real(fft.fft2(fk))

def norm_func(x,fn):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=array([fn(kk) for kk in k])
    fk=fk/p**.5
    return real(fft.fft2(fk.reshape((size,size))))

def bin_by_count(k,n):
    ss={}
    for i in range(1,len(k)):
        if k[i] in ss.keys():
            ss[k[i]].append(i)
        else:
            ss[k[i]]=[i]
    ks=sorted(ss)
    lens=[len(ss[kk]) for kk in ks]
    indices=[]
    currind=[]
    edges=[ks[0]]
    for kk in ks:
        currind = currind + ss[kk]
        if len(currind) >= n:
            edges.append(kk)
            indices.append(currind)
            currind=[]
    indices[-1]=indices[-1]+currind
    edges[-1]=ks[-1]
    return indices,edges

def dynamic_ps(x,count=20):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    indices,edges=bin_by_count(k,count)
    binned_k=[[k[ices] for ices in ind] for ind in indices]
    binned_p=[[p[ices] for ices in ind] for ind in indices]
    mean_p=[mean(pp) for pp in binned_p]
    std_p=[std(pp)/len(pp)**.5 for pp in binned_p]
    mean_k=[mean(kk) for kk in binned_k]
    count = [len(pp) for pp in indices]
    return mean_k,mean_p,std_p,count

def equal_spacing_ps(x,dk):
    size=x.shape[-1]
    k,f = get_flt_k_f(x)
    p=abs(f)**2
    l=int(ceil((size/2+1)*2**.5/dk))
    r=zeros((4,l))
    for i in range(size**2):
        j=int(k[i]/dk)
        r[0,j] += k[i]
        r[1,j] += p[i]
        r[2,j] += p[i]**2
        r[3,j] += 1
    r[0] = r[0]/r[3]
    r[1] = r[1]/r[3]
    r[2] = sqrt((r[2]/r[3]-r[1]**2)/r[3])
    return r

def norm_binning_by_count(x,count=20):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    indices,edges=bin_by_count(k,count)
    binned_k=[[k[ices] for ices in ind] for ind in indices]
    binned_p=[[p[ices] for ices in ind] for ind in indices]
    mean_p=[mean(pp) for pp in binned_p]
    for i,ind in enumerate(indices):
        for ices in ind:
            fk[ices]=fk[ices]/mean_p[i]**.5
    fk[0]=0
    return real(fft.fft2(fk.reshape((size,size))))

def renorm_count(x,fn,count=20):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    indices,edges=bin_by_count(k,count)
    binned_k=[[k[ices] for ices in ind] for ind in indices]
    binned_p=[[p[ices] for ices in ind] for ind in indices]
    mean_p=[mean(pp) for pp in binned_p]
    std_p=[std(pp)/len(pp)**.5 for pp in binned_p]
    for i,ind in enumerate(indices):
        for ices in ind:
            fk[ices]=fk[ices]*(fn(k[ices])/mean_p[i])**.5
    fk[0]=0
    return real(fft.fft2(fk.reshape((size,size))))

def standardise_ps(x, standard, originals=None, count=20, ps=False, interpolate=False):
    n = x.shape[0]
    d = x.ndim-1
    size = x.shape[-1]
    shape = [n]+[size]*(d-1)+[size//2+1]
    factor = size**(d/2)
    axes = tuple(range(1,d+1))
    x = (fft.rfftn(x, axes=axes)/factor).reshape((n,-1))
    f = abs(x)**2
    new = zeros([1]+shape[1:])
    build_ps(new,lambda x: array([standard(x)]), size, d)
    new = new[0].flatten()
    if originals is None:
        k = zeros([1]+shape[1:])
        build_ps(k,lambda x: array([x]), size, d)
        k = k[0].flatten()
        count = 1 if interpolate else count
        indices,edges = bin_by_count(k, count)
        current = zeros(shape).reshape((n,-1))
        nk = len(indices)
        mps = zeros((n,nk))
        for i,ind in enumerate(indices):
            psi = mean(f[:,ind],axis=1)
            current[:,ind] = psi.reshape((-1,1))
            mps[:,i] = psi
        if interpolate:
            knots = array([1.25,2.5,5,10,15,20])/16/2**.5*size/2*d**.5
            splines = []
            for i in range(n):
                splines.append(LSQUnivariateSpline(edges[1:],log(mps[i]),knots))
            current = zeros(shape)
            build_ps(current, lambda k: exp(array([s(k) for s in splines])), size, d)
            current = current.reshape((n,-1))
    else:
        current = originals
    x *= sqrt(new/current)
    x = x.reshape(shape)
    x[tuple([colon]+[0]*d)]=0
    x = fft.irfftn(x,axes=axes)*factor
    if ps and originals is None:
        return x,edges,(splines if interpolate else mps)
    return x

def triangle_plot(samples, labels=None, title=None, nbins=100, bounds=None, expected=None, truth=None, figsize=4):
    n = samples.shape[-1]
    if bounds is None:
        bounds = [(amin(s), amax(s)) for s in samples.T]
    f,ax = subplots(n,n,gridspec_kw={"hspace": 0.05, "wspace": 0.05},figsize=(figsize*n,figsize*n))
    if n==1:
        ax = np.array([[ax]])
    if title is not None:
        suptitle(title)
    bins = [linspace(*bounds[i],nbins+1) for i in range(n)]
    for i in range(n):
        counts = ax[i,i].hist(samples[:,i],bins=bins[i])[0]
        if i<n-1:
            ax[i,i].set_xticks([])
        ax[i,i].set_yticks([])
        ax[i,i].set_xlim(*bounds[i])
        if expected is not None:
            mesh = linspace(*bounds[i], 500)
            expmesh = expected[i](mesh)
            expmesh *= amax(counts)
            ax[i,i].plot(mesh,expmesh,c="k")
        if truth is not None:
            ax[i,i].axvline(truth[i], c="r")
        for j in range(i):
            if i<n-1:
                ax[i,j].set_xticks([])
            if j>0:
                ax[i,j].set_yticks([])
            ax[i,j].hist2d(samples[:,j],samples[:,i], cmap="Blues", bins=[bins[j],bins[i]])
            ax[-(i+1),-(j+1)].axis("off")
            ax[i,j].set_ylim(*bounds[i])
            ax[i,j].set_xlim(*bounds[j])
            if truth is not None:
                ax[i,j].scatter(truth[j],truth[i],c="r",marker="x")
        if labels is not None:
            ax[-1,i].set_xlabel(labels[i])
            if i>0:
                ax[i,0].set_ylabel(labels[i])
        #return f

def corners(samples,bins=30,prenorm=False,binned=False,**kwargs):
    if not iterable(bins):
        bins = linspace(amin(samples),amax(samples),bins+1)
    counts = samples if binned else histogram(samples,bins,**kwargs)[0]
    if prenorm:
        counts = counts/amax(counts)
    x,y = [bins[0]], [0]
    for i in range(len(counts)):
        y += [counts[i], counts[i]]
        x += [bins[i], bins[i+1]]
    x += [bins[-1]]
    y += [0]
    return x,y

def training_fn_generator(x,p,batch_size=128,buffer=1000):
    def f():
        dataset = tf.data.Dataset.from_tensor_slices((x.astype('float32'), p.astype('float32')))
        dataset = dataset.repeat().shuffle(buffer).batch(batch_size)
        return dataset
    return f

def testing_fn_generator(x,p,batch_size=128):
    def f():
        dataset = tf.data.Dataset.from_tensor_slices((x.astype('float32'), p.astype('float32')))
        dataset = dataset.batch(batch_size)
        return dataset
    return f

class LFI(tf.estimator.Estimator):
    def plot_summaries(self, x=None, p=None, testing_fn=None, nbins=100, figsize=4):
        if testing_fn is None:
            testing_fn = testing_fn_generator(x,p)
        stats = array([p["stat"] for p in self.predict(testing_fn)])
        nbands = stats.shape[-1]
        f,ax = subplots(nbands,nbands,gridspec_kw={"hspace": 0.05, "wspace": 0.05},figsize=(figsize*nbands,figsize*nbands))
        if nbands==1:
            ax = array([[ax]])
        suptitle("Summary statistics")
        for i in range(nbands):
            for j in range(nbands):
                ax[i,j].hist2d(p[:,j],stats[:,i],nbins,cmap="Blues")
        for i in range(nbands):
            for j in range(nbands):
                if j==0:
                    ax[i,j].set_ylabel(f"statistic {i+1}")
                else:
                    ax[i,j].set_yticklabels([])
                if i+1 == nbands:
                    ax[i,j].set_xlabel(self.labels[j])
                else:
                    ax[i,j].set_xticklabels([])
        #return f
    
    def plot_means(self, x, p, compare=None, **kwags):
        raise NotImplementedError
        testing_fn = testing_fn_generator(x,p)
        means = [mean(m["samples"], axis=0) for m in self.predict(testing_fn)]
        comparison = [compare(y) for y in x]
                
    
    def plot_posteriors(self, x=None, p=None, compare=None, testing_fn=None, **kwargs):
        if testing_fn is None:
            testing_fn = testing_fn_generator(x,p)
        for i,m in enumerate(self.predict(testing_fn)):
            if compare is not None:
                kwargs["expected"] = compare(x[i])
            yield triangle_plot(m["samples"], labels=self.labels, title=f"sample {i+1}", truth=p[i], **kwargs)
    
    def plot_ranks(self, x=None, p=None, nbins=None, figsize=6, testing_fn=None):
        if testing_fn is None:
            testing_fn = testing_fn_generator(x,p)
        ranks = zeros(p.shape)
        N,n = p.shape
        for i,m in enumerate(self.predict(testing_fn)):
            ranks[i] = sum(m["samples"]<p[i].reshape((1,-1)),axis=0)
        if nbins is None:
            nbins = int(N**.5)
        while self.n_samples%nbins:
            nbins += 1
        f,ax = subplots(n,1,figsize=(figsize,2/3*figsize*n),gridspec_kw={"hspace": 0.3})
        if n==1:
            ax=[ax]
        interval = binom.interval(0.9, N, 1/nbins)
        edges = linspace(0,self.n_samples,nbins+1)
        for i in range(n):
            ax[i].hist(ranks[:,i],edges)
            ax[i].axhspan(*interval, color="r", alpha=0.3)
            ax[i].set_title(self.labels[i])
            ax[i].set_xlabel("rank of truth among posterior samples")
        #return f
    
    def plot_2d_means(self, x, p, bounds=None, cmap=matplotlib.cm.viridis, s=None):
        n = len(x)
        s = 3e4/n if s is None else s
        means = array([mean(p["samples"],axis=0) for p in self.predict(testing_fn_generator(x, p))])
        if bounds is None:
            bounds = zeros((2,2))
            bounds[0] = amin(means, axis=0)
            bounds[1] = amax(means, axis=0)
        if array(bounds).ndim == 1:
            bounds = repeat(array(bounds).reshape(2,1), 2, axis=1)
        f,ax = subplots(2,2,figsize=(12,6),gridspec_kw={"height_ratios": (1,20)})
        norms = [matplotlib.colors.Normalize(vmin=bounds[0,i], vmax=bounds[1,i]) for i in [0,1]]
        colours = [cmap(norms[i](means[:,i])) for i in [0,1]]
        for i in [0,1]:
            ax[1,i].scatter(p[:,0],p[:,1],c=colours[i],s=s,alpha=0.8)
            ax[1,i].set_aspect("equal")
            ax[1,i].set_xlabel(self.labels[0])
            ax[1,i].set_ylabel(self.labels[1])
            ax[0,i].set_title(self.labels[i])
            matplotlib.colorbar.ColorbarBase(ax[0,i],orientation="horizontal",cmap=cmap,norm=norms[i])
        suptitle("Mean posteriors")
        #return f
    
    def plot_2d(self, x, p, bounds=None, cmap=matplotlib.cm.viridis, s=None):
        n = len(x)
        s = 3e4/n if s is None else s
        data = array([[mean(p["samples"],axis=0),p["stat"]] for p in self.predict(testing_fn_generator(x, p))])
        means, stats = data[:,0,:], data[:,1,:]
        if bounds is None:
            bounds = zeros((2,2))
            bounds[0] = amin(means, axis=0)
            bounds[1] = amax(means, axis=0)
        if array(bounds).ndim == 1:
            bounds = repeat(array(bounds).reshape(2,1), 2, axis=1)
        f,ax = subplots(2,4,figsize=(21,6),gridspec_kw={"height_ratios": (1,20)})
        norms = [matplotlib.colors.Normalize(vmin=bounds[0,i], vmax=bounds[1,i]) for i in [0,1]]
        colours = [cmap(norms[i](means[:,i])) for i in [0,1]]
        for i in [0,1]:
            ax[1,i].scatter(p[:,0],p[:,1],c=colours[i],s=s,alpha=0.8)
            ax[1,i].set_aspect("equal")
            ax[1,i].set_xlabel(self.labels[0])
            ax[1,i].set_ylabel(self.labels[1])
            ax[0,i].set_title("mean "+self.labels[i])
            matplotlib.colorbar.ColorbarBase(ax[0,i],orientation="horizontal",cmap=cmap,norm=norms[i])
        norms = [matplotlib.colors.Normalize(vmin=amin(stats[:,i]), vmax=amax(stats[:,i])) for i in [0,1]]
        colours = [cmap(norms[i](stats[:,i])) for i in [0,1]]
        for i in [0,1]:
            ax[1,i+2].scatter(p[:,0],p[:,1],c=colours[i],s=s,alpha=0.8)
            ax[1,i+2].set_aspect("equal")
            ax[1,i+2].set_xlabel(self.labels[0])
            ax[1,i+2].set_ylabel(self.labels[1])
            ax[0,i+2].set_title(f"summary statistic {i+1}")
            matplotlib.colorbar.ColorbarBase(ax[0,i+2],orientation="horizontal",cmap=cmap,norm=norms[i])
    
    def __init__(self,
                feature_columns,
                label_columns,
                n_mixture=10,
                diag=False,
                optimizer=tf.train.AdamOptimizer,
                activation_fn=tf.nn.relu,
                normalizer_fn=tf.contrib.layers.batch_norm,
                dropout=0.2,
                model_dir=None,
                config=None,
                n_samples = 2000,
                model_fn=None,
                cnn=True,
                num_dense=5,
                learning_rate=None,
                kernel_size=2,
                strides=None,
                input_depth=1):
        
        if strides is None:
            strides = kernel_size
        
        if model_fn is not None:
            return tf.estimator.Estimator.__init__(self,model_fn=model_fn,
                                             model_dir=model_dir,
                                             config=config)
        
        if learning_rate is None:
            learning_rate = lambda x: tf.train.exponential_decay(0.001, x, 1000, 0.7, staircase=False)

        def _model_fn(features, labels, mode):
            label_dimension = len(label_columns)
            training=mode == tf.estimator.ModeKeys.TRAIN
            # Builds the neural network
            size = int(features.shape[-1])
            d = len(features.shape) - 1 - (input_depth>1)
            #print(size,d,input_depth)
            #assert not cnn or 1<=d<=3
            if cnn:
                if input_depth == 1:
                    conv = tf.expand_dims(features,-1)
                else:
                    conv = features
                channels = input_depth
                width = size
                conv_layer = [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d][d-1]
                while width > 1:
                    channels *= kernel_size**d
                    channels = min(channels, 1024)
                    #print(width,channels)
                    conv = conv_layer(conv, channels, kernel_size, strides=strides, activation=tf.nn.leaky_relu)
                    width = conv.shape[-2]
                    #tf.print(width,type(width))
            else:
                channels = size**d*input_depth
                conv = features
            dense = tf.reshape(conv,(-1,channels))
            f = -int(-(channels/label_dimension)**(1/num_dense))
            for i in range(num_dense-1):
                channels //= f
                #print(channels)
                dense = tf.contrib.layers.fully_connected(tf.layers.dropout(dense,rate=dropout,training=training),channels,activation_fn=tf.nn.leaky_relu)
            stat = tf.contrib.layers.fully_connected(tf.layers.dropout(dense,rate=dropout,training=training),label_dimension)
            
            net = tf.contrib.layers.fully_connected(stat, 128, activation_fn=tf.nn.tanh)
            
                # Size of the covariance matrix
            if diag ==True:
                size_sigma = label_dimension
            else:
                size_sigma = (label_dimension *(label_dimension +1) // 2)

            # Create mixture components from network output
            out_mu = tf.contrib.layers.fully_connected(net, label_dimension*n_mixture , activation_fn=None)
            out_mu = tf.reshape(out_mu, (-1, n_mixture, label_dimension))

            out_sigma = tf.contrib.layers.fully_connected(net, size_sigma * n_mixture, activation_fn=None)
            out_sigma = tf.reshape(out_sigma, (-1, n_mixture, size_sigma))

            out_p = tf.contrib.layers.fully_connected(net, n_mixture, activation_fn=None)

            if diag == True:
                sigma_mat = tf.nn.softplus(out_sigma)+1e-4
                gmm = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=out_p),
                              components_distribution=tfp.distributions.MultivariateNormalDiag(loc=out_mu,
                                                                                scale_diag=sigma_mat))
            else:
                sigma_mat = tfp.distributions.matrix_diag_transform(tfp.distributions.fill_triangular(out_sigma), transform=tf.nn.softplus)
                gmm = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=out_p),
                             components_distribution=tfp.distributions.MultivariateNormalTriL(loc=out_mu,
                                                                                scale_tril=sigma_mat))

            predictions = {'mu': out_mu, 'sigma': sigma_mat, 'p':out_p, 'stat':stat}

            if mode == tf.estimator.ModeKeys.PREDICT:
                y = gmm.sample(n_samples)
                predictions['samples'] = tf.transpose(y,[1,0,2])

                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                                  export_outputs={'pdf': tf.estimator.export.PredictOutput(predictions),
                                                                  'samples': tf.estimator.export.PredictOutput(y),
                                                                  tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(y)})
            label_layer = labels

            # Compute and register loss function
            loss = - tf.reduce_mean(gmm.log_prob(label_layer),axis=0)
            tf.losses.add_loss(loss)
            total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

            train_op = None
            eval_metric_ops = None
            
            # Define optimizer
            if mode == tf.estimator.ModeKeys.TRAIN:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                lr = learning_rate(tf.train.get_global_step()) if callable(learning_rate) else learning_rate
                with tf.control_dependencies(update_ops):
                    train_op = optimizer(learning_rate=lr).minimize(loss=total_loss,
                                                global_step=tf.train.get_global_step())
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('learning rate', lr)
            elif mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = { "log_p": loss}

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              loss=total_loss,
                                              train_op=train_op,
                                              eval_metric_ops=eval_metric_ops)

        tf.estimator.Estimator.__init__(self,model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)
        self.labels = label_columns
        self.n_samples = n_samples
