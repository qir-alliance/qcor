![qcor](docs/assets/qcor_full_logo.svg)

| master | 
|:-------|
| [![pipeline status](https://code.ornl.gov/qci/qcor/badges/master/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/master) |

# QCOR

QCOR is a C++ language extension and associated compiler implementation
for hybrid quantum-classical programming.

## Documentation

* [Documentation and User Guides](https://aide-qc.github.io/deploy)
* [Doxygen API Docs](https://ornl-qci.github.io/qcor-api-docs/)

## Installation
To install the `qcor` nightly binaries (for Mac OS X and Linux x86_64) run the following command from your terminal 
```bash
/bin/bash -c "$(curl -fsSL https://aide-qc.github.io/deploy/install.sh)"
```
To use the Python API, be sure to set your `PYTHONPATH`. 
For more details, see the [full installation documentation page](https://aide-qc.github.io/deploy/getting_started/).

### Docker Images

Nightly docker images are also available that serve up a [VSCode IDE](https://github.com/cdr/code-server) on port 8080. To use this image, run 
```bash
docker run -it -p 8080:8080 qcor/qcor
```
and navigate to ``https://localhost:8080`` in your browser to open the IDE and get started with QCOR. 

Alternatively, you could use the `qcor/cli` image providing simple command-line access to the `qcor` compiler. 
```bash
docker run -it qcor/cli
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
