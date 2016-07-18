pyfigtree
=========

A python ctypes wrapper of the
[figtree library](https://github.com/vmorariu/figtree) for fast
Gaussian summation by V. Morariu et al.

The main function for users is `pyfigtree.figtree`. It computes the
improved fast Gauss transform

    g(y) = \sum_{i=1}^N w_i \exp( -|x_i - y|^2 / h^2)

for N samples `{x_i}` at the target point y.

Kernel density estimation
-------------------------

For a properly normalized Gaussian kernel density estimation in 1D,
the weight is

    w_i = 1 / (N \sqrt{\pi h^2}),

where `h` is the *bandwidth*.  Details about the algorithm and the
parameters are given in the
[original paper](http://papers.nips.cc/paper/3420-automatic-online-tuning-for-fast-gaussian-summation.pdf).

Note that multidimensional input usually has to be transformed to
avoid distortions if the variates are of different scales.  The
fastest strategy is to scale the samples to the unit hypercube. For example in 2D

```python
for i in range(2):
    x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())
```

If that is not good enough (i.e., scales still too different), transform
into *almost principal components* as suggested in [Scott, Sain (1992),
section 3.3](http://bama.ua.edu/~mdporter2/papers/Multi-dimensional%20density%20estimation_Scott_Sain.pdf).

Example
-------

Sample from a unit Gaussian, and do kernel density estimation with
figtree. The weights are adjusted such that the density is normalized
correctly.

```python
from figtree import figtree
import numpy as np

samples = np.random.normal(size=1000)
bandwidth = 0.5
weights = np.ones(len(samples)) / len(samples) / np.sqrt(np.pi) / bandwidth
target_points = np.linspace(-5, 5, 70)
target_densities = figtree(samples, target_points, weights, bandwidth=bandwidth)

from matplotlib import pyplot as plt

plt.plot(target_points, target_densities)
plt.hist(samples, histtype='stepfilled', normed=True)
plt.show()
```

Installation
------------

This wrapper has been developed and tested only on linux. To use it,
first install both the figtree and the ANN library following the
instructions at https://github.com/vmorariu/figtree and make the
libraries available to the loader at runtime. For example,

```sh
export FIGTREEDIR=/path/to/figtree
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FIGTREEDIR/lib
export PYTHONPATH=/path/to/pyfigtree:$PYTHONPATH
```

Then

* add `pyfigtree.py` to your `PYTHON_PATH`;
* make sure `numpy` is installed;
* test the setup by executing `python figtree.py` to run a set of unit
  tests.

Historical note
---------------

I wrote this wrapper around 2011. In 2014, I figured I should clean it
up and release it on github because it may be useful to others. When I
notified Vlad Morariu, the author of figtree, he told me that he had
in fact done the same so now there are two
[python wrappers](https://github.com/vmorariu/figtree#python-wrapper).

What's the difference? Vlad uses `cython` and I use `ctypes`; the
latter is included in a standard `python` installation, so I have one
less dependency and I don't need to compile anything, the only requirement
is to be able to load the figtree libraries at runtime.

License
-------

Copyright (c) 2014 Frederik Beaujean <Frederik.Beaujean@lmu.de>

Pyfigtree is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License version 2, as
published by the Free Software Foundation.

This software is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
