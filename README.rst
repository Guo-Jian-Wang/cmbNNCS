cmbNNCS
=======

**cmbNNCS (CMB Neural Network Component Separator)**

cmbNNCS is a method for component separation of the cosmic microwave background (CMB) observations using convolutional neural network (CNN).

It is proposed by `Guo-Jian Wang, Hong-Liang Shi, Ye-Peng Yan, Jun-Qing Xia, Yan-Yun Zhao, Si-Yu Li, Jun-Feng Li (2022) <https://arxiv.org/abs/2204.01820>`_.



Attribution
-----------

If you use this code in your research, please cite `Guo-Jian Wang, Hong-Liang Shi, Ye-Peng Yan, et al. 2022, ApJS, XXX, XXX <https://arxiv.org/abs/2204.01820>`_.



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
* `coplot <https://github.com/Guo-Jian-Wang/coplot>`_



Installation
------------

You can install cmbnncs by using::

    $ git clone https://github.com/Guo-Jian-Wang/cmbnncs.git    
    $ cd cmbnncs
    $ sudo python setup.py install



Usage
-----

There are two main parts in the code: generating the training (test) set and training the CNN model.

The files sim_*.py in the examples folder are used to simulate the training (test) set. The files add_*py are used to add instrument noise and beam effects. However, sim_*py are using a modified version of PySM to generate the mock data, which is not included in this code. Therefore, we recommend the interested readers simulate CMB and foreground maps using the original `PySM <https://github.com/bthorne93/PySM_public>`_.

The files train_*.py and test_*.py in the examples folder are used for the training and testing of the CNN model, respectively.



Contributors
------------

Guo-Jian Wang

Si-Yu Li


License
-------

Copyright 2022-2022 Guojian Wang

cmbnncs is free software made available under the MIT License. For details see the LICENSE file.
