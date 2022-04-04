cmbNNCS
=======

**cmbNNCS (CMB Neural Network Component Separator)**

cmbNNCS is a method for component separation of the cosmic microwave background (CMB) observations using convolutional neural network (CNN).

It is proposed by `Guo-Jian Wang, Hong-Liang Shi, Ye-Peng Yan, et al. (2022) <>`_.


Attribution
-----------


Dependencies
------------

The main dependencies of cmbnncs are:

* `PyTorch <https://pytorch.org/>`_
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_
* `PySM <https://github.com/bthorne93/PySM_public>`_
* `CAMB <https://github.com/cmbant/CAMB>`_
* `Healpy <https://github.com/healpy/healpy>`_
* `NaMaster <https://github.com/LSSTDESC/NaMaster>`_
* `Astropy <https://github.com/astropy/astropy>`_

and some commonly used modules:

* os
* sys
* numpy
* scipy
* math
* shutil
* matplotlib
* itertools
* collections
* time


Installation
------------

You can install cmbnncs by using::

    $ git clone https://github.com/Guo-Jian-Wang/cmbnncs.git    
    $ cd cmbnncs
    $ sudo python setup.py install


Contributors
------------

Guo-Jian Wang

Si-Yu Li


License
-------

Copyright 2022-2022 Guojian Wang

cmbnncs is free software made available under the MIT License. For details see the LICENSE file.
