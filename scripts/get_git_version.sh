#!/bin/bash
git ls-remote https://github.com/ornl-qci/qcor HEAD | awk '{ print $1}' | head -c 7