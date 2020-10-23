from _pyqcor import *
# TODO: will need to selectively import XACC
import xacc
import sys
from typing import Union, List


def X(idx):
    return xacc.quantum.PauliOperator({idx: 'X'}, 1.0)


def Y(idx):
    return xacc.quantum.PauliOperator({idx: 'Y'}, 1.0)


def Z(idx):
    return xacc.quantum.PauliOperator({idx: 'Z'}, 1.0)


class qjit(object):
    def __init__(self, function, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.allowed_type_cpp_map = {'<class \'_pyqcor.qreg\'>': 'qreg',
                                     '<class \'float\'>': 'double', 'typing.List[float]': 'std::vector<double>'}
        self.__dict__.update(kwargs)
        return

    def __call__(self, *args, **kwargs):
        import inspect
        # Get the function body source as a string
        fbody_src = '\n'.join(inspect.getsource(self.function).split('\n')[2:])

        # Get the arg names and type annotation
        arg_names, _, _, _, _, _, type_annotations = inspect.getfullargspec(
            self.function)
        if not type_annotations:
            print('Error, you must provide type annotations for qcor quantum kernels.')
            exit(1)

        # Construct the C++ kernel arg string
        cpp_arg_str = ''
        for arg, _type in type_annotations.items():
            if str(_type) not in self.allowed_type_cpp_map:
                print('Error, this quantum kernel arg type is not allowed: ', str(_type))
                exit(1)
            cpp_arg_str += ',' + \
                self.allowed_type_cpp_map[str(_type)] + ' ' + arg
        cpp_arg_str = cpp_arg_str[1:]

        # Update as a qcor quantum kernel function for QJIT
        fbody_src = '__qpu__ void '+self.function.__name__ + \
            '('+cpp_arg_str+') {\nusing qcor::pyxasm;\n'+fbody_src+"}\n"

        # Run JIT, this will reused cached JIT LLVM Modules
        _qjit = QJIT()
        _qjit.jit_compile(fbody_src)

        # Create a dictionary for the function arguments
        args_dict = {}
        for i, arg_name in enumerate(arg_names):
            args_dict[arg_name] = list(args)[i]

        # Invoke the JITed function
        _qjit.invoke(self.function.__name__, args_dict)

        return

# Must have qpu in the init kwargs
# defaults to qpp, but will search for -qpu flag
init_kwargs = {'qpu': sys.argv[sys.argv.index('-qpu')+1] if '-qpu' in sys.argv else 'qpp'}

# get shots if provided
if '-shots' in sys.argv:
    init_kwargs['shots'] = int(sys.argv[sys.argv.index('-shots')+1])

# Implements internal_startup initialization:
# i.e. set up qrt, backends, shots, etc.
Initialize(**init_kwargs)
