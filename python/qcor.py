import xacc
from _pyqcor import *
import sys, inspect
from typing import List
import typing 

List = typing.List

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

        self._qjit = QJIT()

        fbody_src = '\n'.join(inspect.getsource(self.function).split('\n')[2:])

        self.arg_names, _, _, _, _, _, self.type_annotations = inspect.getfullargspec(
            self.function)
        if not self.type_annotations or len(self.arg_names) != len(self.type_annotations):
            print('Error, you must provide type annotations for qcor quantum kernels.')
            exit(1)

        # Construct the C++ kernel arg string
        cpp_arg_str = ''
        for arg, _type in self.type_annotations.items():
            if str(_type) not in self.allowed_type_cpp_map:
                print('Error, this quantum kernel arg type is not allowed: ', str(_type))
                exit(1)
            cpp_arg_str += ',' + \
                self.allowed_type_cpp_map[str(_type)] + ' ' + arg
        cpp_arg_str = cpp_arg_str[1:]

        # Update as a qcor quantum kernel function for QJIT
        self.src = '__qpu__ void '+self.function.__name__ + \
            '('+cpp_arg_str+') {\nusing qcor::pyxasm;\n'+fbody_src+"}\n"

        self._qjit.internal_python_jit_compile(self.src)

        return

    def get_internal_src(self):
        return self.src

    def kernel_name(self):
        return self.function.__name__

    def translate(self, q: qreg, x: List[float]):
        if [str(x) for x in self.type_annotations.values()] == ['<class \'_pyqcor.qreg\'>', '<class \'float\'>']:
            ret_dict = {}
            for arg_name, _type in self.type_annotations.items():
                if str(_type) == '<class \'_pyqcor.qreg\'>':
                    ret_dict[arg_name] = q
                elif str(_type) == '<class \'float\'>':
                    ret_dict[arg_name] = x[0]
            if len(ret_dict) != len(self.type_annotations):
                print('Error, could not translate vector parameters x into arguments for quantum kernel.')
                exit(1)
            return ret_dict
        elif [str(x) for x in self.type_annotations.values()] == ['<class \'_pyqcor.qreg\'>', 'typing.List[float]']:
            ret_dict = {}
            for arg_name, _type in self.type_annotations.items():
                if str(_type) == '<class \'_pyqcor.qreg\'>':
                    ret_dict[arg_name] = q
                elif str(_type) == 'typing.List[float]':
                    ret_dict[arg_name] = x
            if len(ret_dict) != len(self.type_annotations):
                print('Error, could not translate vector parameters x into arguments for quantum kernel.')
                exit(1)
            return ret_dict
        else:
            print('currently cannot translate other arg structures')
            exit(1)

    def extract_composite(self, *args):
        # Create a dictionary for the function arguments
        args_dict = {}
        for i, arg_name in enumerate(self.arg_names):
            args_dict[arg_name] = list(args)[i]

        return self._qjit.extract_composite(self.function.__name__, args_dict)

    def __call__(self, *args):
        # Create a dictionary for the function arguments
        args_dict = {}
        for i, arg_name in enumerate(self.arg_names):
            args_dict[arg_name] = list(args)[i]

        # Invoke the JITed function
        self._qjit.invoke(self.function.__name__, args_dict)

        return


# Must have qpu in the init kwargs
# defaults to qpp, but will search for -qpu flag
init_kwargs = {'qpu': sys.argv[sys.argv.index(
    '-qpu')+1] if '-qpu' in sys.argv else 'qpp'}

# get shots if provided
if '-shots' in sys.argv:
    init_kwargs['shots'] = int(sys.argv[sys.argv.index('-shots')+1])

# Implements internal_startup initialization:
# i.e. set up qrt, backends, shots, etc.
Initialize(**init_kwargs)
