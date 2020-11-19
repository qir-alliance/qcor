![qcor](docs/assets/qcor_full_logo.svg)

| master | 
|:-------|
| [![pipeline status](https://code.ornl.gov/qci/qcor/badges/master/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/master) |

# QCOR

QCOR is a C++ language extension and associated compiler implementation
for hybrid quantum-classical programming.


Documentation
-------------

* [Website and Documentation](https://aide-qc.github.io/deploy)
* [Doxygen Documentation](https://ornl-qci.github.io/qcor-api-docs/)

Install
-------
To install `qcor` run the following command from your terminal 
```bash
/bin/bash -c "$(curl -fsSL https://aide-qc.github.io/deploy/install.sh)"
```
For more details, see [here](https://aide-qc.github.io/deploy/getting_started/).

Nightly docker images are also available that serve up a [Theia IDE](https://theia-ide.org/) on port 3000. To use this image, run 
```bash
docker run --security-opt seccomp=unconfined --init -it -p 3000:3000 qcor/qcor
```
and navigate to ``https://localhost:3000`` in your browser to open the IDE and get started with QCOR. 

For any method of installation, a good way to test your install is to copy and paste the following into your terminal 
```bash
printf "__qpu__ void f(qreg q) {
  H(q[0]);
  Measure(q[0]);
}
int main() {
  auto q = qalloc(1);
  f(q);
  q.print();
}  " | qcor -qpu qpp -shots 1024 -o test -x c++ -
```
and then run 
```bash
./test
```

## Cite QCOR 
If you use qcor in your research, please use the following citation 
```
@ARTICLE{qcor,
       author = {{Nguyen}, Thien and {Santana}, Anthony and {Kharazi}, Tyler and
         {Claudino}, Daniel and {Finkel}, Hal and {McCaskey}, Alexander},
        title = "{Extending C++ for Heterogeneous Quantum-Classical Computing}",
      journal = {arXiv e-prints},
     keywords = {Quantum Physics, Computer Science - Mathematical Software},
         year = 2020,
        month = oct,
          eid = {arXiv:2010.03935},
        pages = {arXiv:2010.03935},
archivePrefix = {arXiv},
       eprint = {2010.03935},
 primaryClass = {quant-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201003935N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```