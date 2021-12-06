#!/bin/bash
git ls-remote https://github.com/qir-alliance/qcor HEAD | awk '{ print $1}' | head -c 7