Installation
============

Quick-Start with Docker
-----------------------
To get up and running quickly and avoid installing the prerequisites you can
pull the ``qcor/qcor`` Docker image.
This image provides an Ubuntu 18.04 container that serves up an Eclipse Theia IDE. QCOR is already
built and ready to go. 

Dependencies
------------
Note that you must have a C++17 compliant compiler and a recent version of CMake (version 3.12+). 
You must have XACC installed (see `Bulding XACC <https://xacc.readthedocs.io/en/latest/install.html#building-xacc>`_)

Easiest way to install CMake - do not use the package manager,
instead use `pip`, and ensure that `/usr/local/bin` is in your PATH:

.. code:: bash

   $ python3 -m pip install --upgrade cmake
   $ export PATH=$PATH:/usr/local/bin

For now we require our users build a specific fork of LLVM/Clang that 
provides Syntax Handler plugin support. We expect this fork to be upstreamed 
in a future release of LLVM and Clang, and at that point users will only 
need to download the appropriate LLVM/Clang binaries (via `apt-get` for instance).

To build this fork of LLVM/Clang (be aware this step takes up a good amount of RAM):

.. code:: bash

   $ apt-get install ninja-build [if you dont have ninja]
   $ git clone https://github.com/hfinkel/llvm-project-csp llvm
   $ cd llvm && mkdir build && cd build
   $ cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=$HOME/.llvm -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_PROJECTS=clang
   $ cmake --build . --target install
   $ sudo ln -s $HOME/.llvm/bin/llvm-config /usr/bin

Building from Scratch
---------------------

Note that, for now, developers must clone QCOR manually:

.. code:: bash 

   $ git clone https://github.com/ornl-qci/qcor
   $ cd qcor && mkdir build && cd build
   $ cmake .. 
   $ [with tests] cmake .. -DQCOR_BUILD_TESTS=TRUE
   $ make -j$(nproc) install

Update your PATH to ensure that the ```qcor``` compiler is available.

.. code:: bash

   $ export PATH=$PATH:$HOME/.xacc/bin (or wherever you installed XACC)

