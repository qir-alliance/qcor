![qcor](docs/assets/qcor_full_logo.svg)

| master | 
|:-------|
| [![pipeline status](https://github.com/qir-alliance/qcor/actions/workflows/ci-linux.yml/badge.svg?branch=master)](https://github.com/qir-alliance/qcor/actions/workflows/ci-linux.yml) |

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

## Feedback

If you have feedback about the content in this repository, please let us know by
filing a [new issue](https://github.com/qir-alliance/qcor/issues/new)!

## Contributing

There are many ways in which you can contribute to QCOR, whether by contributing
a feature or by engaging in discussions; we value contributions in all shapes
and sizes! We refer to [this document](CONTRIBUTING.md) for guidelines and ideas
for how you can get involved.

Contributing a pull request to this repo requires to agree to a
[Contributor License Agreement (CLA)](https://en.wikipedia.org/wiki/Contributor_License_Agreement)
declaring that you have the right to, and actually do, grant us the rights to
use your contribution. We are still working on setting up a suitable CLA-bot to
automate this process. A CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately. Once it is set up, simply
follow the instructions provided by the bot. You will only need to do this once.

## Code of Conduct

This project has adopted the community covenant
[Code of Conduct](https://github.com/qir-alliance/.github/blob/main/Code_of_Conduct.md#contributor-covenant-code-of-conduct).
Please contact [qiralliance@mail.com](mailto:qiralliance@mail.com) for Code of
Conduct issues or inquires.