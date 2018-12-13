## Getting started with the development of IR Transformations

Developers can take advantage of the XACC plugin generator to create 
all the base boilerplate code required to inject new IR Transformations 
into the framework. From this directory, run the following:

```bash
$ python3 -m xacc generate-plugin -t irtransformation -n example
```

If you now look in the folder that was generated, you will see an 
transformation subfolder. In that folder will be a stubbed out version 
of your new IR Transformation.

To build and test
```bash
$ mkdir build && cd build
$ cmake .. -DEXAMPLE_BUILD_TESTS=TRUE
$ make -j4
$ ctest
```

We will work on getting the IR Transformation integrated with the 
qcor build system later.