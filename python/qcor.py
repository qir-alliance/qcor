from _pyqcor import *
# TODO: will need to selectively import XACC
import xacc


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
        self.__dict__.update(kwargs)
        return

    def __call__(self, *args, **kwargs):
        import inspect
        print('calling here, ', self.function,
              inspect.getsource(self.function))
        fbody_src = '\n'.join(inspect.getsource(self.function).split('\n')[2:])
        # src = src[1:]
        fbody_src = '__qpu__ void '+self.function.__name__ + \
            '(qreg q) {\nusing qcor::pyxasm;\n'+fbody_src+"}\n"
        print(fbody_src)

        _qjit = QJIT()
        _qjit.jit_compile(fbody_src)

        args_dict = {}
        for i, arg_name in enumerate(*inspect.getfullargspec(self.function)[0]):
            args_dict[arg_name] = list(args)[i]

        _qjit.invoke(self.function.__name__, args_dict)

        return


qrt_initialize('qpp', 'tmp')
