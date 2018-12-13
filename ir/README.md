## Getting started with the development of analog instructions

Developers can take advantage of the XACC plugin generator to create 
all the base boilerplate code required to inject new Instructions 
into the framework. From this directory, run the following:

```bash
$ python3 -m xacc generate-plugin -t gate-instruction -n analogexample
```

If you now look in the folder that was generated, you will see an 
instruction subfolder. In that folder will be a stubbed out version 
of your new Instruction (analog or digital depending on what you want 
to implement). 

If you are describing an analog instruction, you may not want to 
implement the ``describeOptions`` method. There is an example commented out. 
You can implement unit tests for your new instruction in the tests/ folder, 
and there is stubbed out code for that as well. 

To build and test
```bash
$ mkdir build && cd build
$ cmake .. -DANALOGEXAMPLE_BUILD_TESTS=TRUE
$ make -j4
$ ctest
```

We will work on getting the instruction integrated with the 
qcor build system later.