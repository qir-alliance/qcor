import sys, uuid, atexit, hashlib

if '@QCOR_APPEND_PLUGIN_PATH@':
    sys.argv += ['__internal__add__plugin__path', '@QCOR_APPEND_PLUGIN_PATH@']

import xacc 

from _pyqcor import *
import inspect, ast
from typing import List
import typing, types
import re
import itertools
from collections import defaultdict

List = typing.List
Tuple = typing.Tuple
MethodType = types.MethodType
Callable = typing.Callable

# KernelSignature type annotation:
# Usage: annotate an function argument as a KernelSignature by:
# varName: KernelSignature(qreg, ...)
# Kernel always returns void (None)
def KernelSignature(*args):
    return Callable[list(args), None]

# Static cache of all Python QJIT objects that have been created.
# There seems to be a bug when a Python interpreter tried to create a new QJIT
# *after* a previous QJIT is destroyed.
# Note: this could only occur when QJIT kernels were declared in local scopes.
# i.e. multiple kernels all declared in global scope don't have this issue.
# Hence, to be safe, we cache all the QJIT objects ever created until QCOR module is unloaded.
QJIT_OBJ_CACHE = []
@atexit.register
def clear_qjit_cache():
    QJIT_OBJ_CACHE = []

PauliOperator = xacc.quantum.PauliOperator
FermionOperator = xacc.quantum.FermionOperator 
FLOAT_REF = typing.NewType('value', float)
INT_REF = typing.NewType('value', int)

typing_to_simple_map = {'<class \'_pyqcor.qreg\'>': 'qreg',
                            '<class \'_pyqcor.qubit\'>': 'qubit',
                            '<class \'float\'>': 'float', 'typing.List[float]': 'List[float]',
                            '<class \'int\'>': 'int', 'typing.List[int]': 'List[int]',
                            '<class \'_pyxacc.quantum.PauliOperator\'>': 'PauliOperator',
                            '<class \'_pyxacc.quantum.FermionOperator\'>': 'FermionOperator',
                            'typing.List[typing.Tuple[int, int]]': 'List[Tuple[int,int]]',
                            'typing.List[_pyxacc.quantum.PauliOperator]': 'List[PauliOperator]',
                            'typing.List[_pyxacc.quantum.FermionOperator]': 'List[FermionOperator]'}

# Need to add a few extra header paths 
# for the clang code-gen mechanism. Mac OS X will 
# need QCOR_EXTRA_HEADERS, all will need the 
# Python include path.
extra_headers = ['-I'+'@Python_INCLUDE_DIRS@']
tmp_extra_headers = '@QCOR_EXTRA_HEADERS@'.replace('"','')
for path in tmp_extra_headers.split(';'):
    if path:
        extra_headers.append('-I'+path)

def X(idx):
    return xacc.quantum.PauliOperator({idx: 'X'}, 1.0)


def Y(idx):
    return xacc.quantum.PauliOperator({idx: 'Y'}, 1.0)


def Z(idx):
    return xacc.quantum.PauliOperator({idx: 'Z'}, 1.0)

def adag(idx):
    return xacc.quantum.FermionOperator([(idx,True)], 1.0)

def a(idx):
    return xacc.quantum.FermionOperator([(idx,False)], 1.0)

cpp_matrix_gen_code = '''#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
namespace py = pybind11;
// returns 1d data as vector and matrix size (assume square)
auto __internal__qcor_pyjit_gen_{}_unitary_matrix({}) {{
  auto py_src = R"#({})#";
  auto locals = py::dict();
  {}
  py::exec(py_src, py::globals(), locals);
  return std::make_pair(
      locals["mat_data"].cast<std::vector<std::complex<double>>>(), 
      locals["mat_size"].cast<int>());
}}'''

# Simple graph class to help resolve kernel dependency (via topological sort)


class KernelGraph(object):
    def __init__(self):
        self.graph = defaultdict(list)
        self.V = 0
        self.kernel_idx_dep_map = {}
        self.kernel_name_list = []

    def addKernelDependency(self, kernelName, depList):
        self.kernel_name_list.append(kernelName)
        self.kernel_idx_dep_map[self.V] = []
        for dep_ker_name in depList:
            self.kernel_idx_dep_map[self.V].append(
                self.kernel_name_list.index(dep_ker_name))
        self.V += 1

    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Topological Sort.
    def topologicalSort(self):
        self.graph = defaultdict(list)
        for sub_ker_idx in self.kernel_idx_dep_map:
            for dep_sub_idx in self.kernel_idx_dep_map[sub_ker_idx]:
                self.addEdge(dep_sub_idx, sub_ker_idx)

        in_degree = [0]*(self.V)
        for i in self.graph:
            for j in self.graph[i]:
                in_degree[j] += 1

        queue = []
        for i in range(self.V):
            if in_degree[i] == 0:
                queue.append(i)
        cnt = 0
        top_order = []
        while queue:
            u = queue.pop(0)
            top_order.append(u)
            for i in self.graph[u]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
            cnt += 1

        sortedDep = []
        for sorted_dep_idx in top_order:
            sortedDep.append(self.kernel_name_list[sorted_dep_idx])
        return sortedDep

    def getSortedDependency(self, kernelName):
        kernel_idx = self.kernel_name_list.index(kernelName)
        # No dependency
        if len(self.kernel_idx_dep_map[kernel_idx]) == 0:
            return []

        sorted_dep = self.topologicalSort()
        result_dep = []
        for dep_name in sorted_dep:
            if dep_name == kernelName:
                return result_dep
            else:
                result_dep.append(dep_name)


class qjit(object):
    """
    The qjit class serves a python function decorator that enables 
    the just-in-time compilation of quantum python functions (kernels) using 
    the QCOR QJIT infrastructure. Example usage:

    @qjit
    def kernel(qbits : qreg, theta : float):
        X(q[0])
        Ry(q[1], theta)
        CX(q[1], q[0])
        for i in range(q.size()):
            Measure(q[i])

    q = qalloc(2)
    kernel(q)
    print(q.counts())

    Upon initialization, the python inspect module is used to extract the function body 
    as a string. This string is processed to create a corresponding C++ function with 
    pythonic function body as an embedded domain specific language. The QCOR QJIT engine 
    takes this function string, and delegates to the QCOR Clang SyntaxHandler infrastructure, which 
    maps this function to a QCOR QuantumKernel sub-type, compiles to LLVM bitcode, caches that bitcode 
    for future fast lookup, and extracts function pointers using the LLVM JIT engine that can be called 
    later, affecting execution of the quantum code. 

    Note that kernel function arguments must provide type hints, and allowed types are int, bool, float, List[float], and qreg. 

    qjit annotated functions can also be passed as general functors to other QCOR API calls like 
    createObjectiveFunction, and createModel from the QSim library. 

    """

    def __init__(self, function, *args, **kwargs):
        """Constructor for qjit, takes as input the annotated python function and any additional optional
        arguments that are used to customize the workflow."""
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.allowed_type_cpp_map = {'<class \'_pyqcor.qreg\'>': 'qreg',
                                     '<class \'_pyqcor.qubit\'>': 'qubit',
                                     '<class \'float\'>': 'double', 'typing.List[float]': 'std::vector<double>',
                                     '<class \'int\'>': 'int', 'typing.List[int]': 'std::vector<int>',
                                     '<class \'_pyxacc.quantum.PauliOperator\'>': 'qcor::PauliOperator',
                                     '<class \'_pyxacc.quantum.FermionOperator\'>': 'qcor::FermionOperator',
                                     'typing.List[typing.Tuple[int, int]]': 'PairList<int>',
                                     'typing.List[_pyxacc.quantum.PauliOperator]': 'std::vector<qcor::PauliOperator>',
                                     'typing.List[_pyxacc.quantum.FermionOperator]': 'std::vector<qcor::FermionOperator>'}
        self.__dict__.update(kwargs)

        # Create the qcor just in time engine
        self._qjit = QJIT()
        self.extra_cpp_code = ''

        # Get the kernel function body as a string
        if '__internal_fbody_src_provided__' in kwargs:
            fbody_src = kwargs['__internal_fbody_src_provided__']
        else:
            fbody_src = '\n'.join(inspect.getsource(self.function).split('\n')[2:])

        # Get the arg variable names and their types
        self.arg_names, _, _, _, _, _, self.type_annotations = inspect.getfullargspec(
            self.function)

        # Look at fbody_src, if with decompose is in there, then we
        # want to rewrite that portion to C++ here, that would be easiest.
        # strategy is going to be to run the decompose body code, get the
        # matrix as a 1d array, and rewrite to read it into UnitaryMatrix
        if 'with decompose' in fbody_src:
            # split the function into lines
            lines = fbody_src.split('\n')

            # Get all lines that are 'with decompose...'
            with_decomp_lines = [
                line for line in lines if 'with decompose' in line if line.lstrip()[0] != '#']
            # Get their index in the lines list
            with_decomp_lines_idxs = [
                lines.index(s) for s in with_decomp_lines]
            # Get their column start integer
            with_decomp_lines_col_starts = [
                sum(1 for _ in itertools.takewhile(str.isspace, s)) for s in with_decomp_lines]
            # Get the name of the matrix we are decomposing
            with_decomp_matrix_names = [line.split(
                ' ')[-1][:-1] for line in with_decomp_lines]

            # Loop over all decompose segments
            for i, line_idx in enumerate(with_decomp_lines_idxs):
                stmts_to_run = []
                total_decompose_code = with_decomp_lines[i]
                # Get all lines in the with decompose scope
                # ends if we hit a line with column dedent
                for line in lines[line_idx+1:]:
                    col_loc = sum(
                        1 for _ in itertools.takewhile(str.isspace, line))
                    if col_loc == with_decomp_lines_col_starts[i]:
                        break
                    total_decompose_code += '\n' + line
                    stmts_to_run.append(line.lstrip())

                # Get decompose args
                decompose_args = re.search(
                    '\(([^)]+)', with_decomp_lines[i]).group(1)
                d_list = decompose_args.split(',')
                decompose_args = d_list[0]
                for e in d_list[1:]:
                    if 'depends_on' in e:
                        break
                    decompose_args += ',' + e

                # Build up the matrix generation code
                code_to_exec = 'import numpy as np\n' + \
                    '\n'.join([s for s in stmts_to_run])
                code_to_exec += '\nmat_data = np.array(' + \
                    with_decomp_matrix_names[i]+').flatten()\n'
                code_to_exec += 'mat_size = ' + \
                    with_decomp_matrix_names[i]+'.shape[0]\n'
                # Users can use numpy. or np.
                code_to_exec = code_to_exec.replace('numpy.', 'np.')

                # Figure out it code_to_exec depends on any 
                # kernel arguments
                class FindDependentKernelVariables(ast.NodeVisitor):
                    def __init__(self, arg_names):
                        self.depends_on = []
                        self.outer_parent_arg_names = arg_names
                    def visit_Name(self, node):
                        if node.id in self.outer_parent_arg_names:
                            self.depends_on.append(node.id)
                        self.generic_visit(node)
                tree = ast.parse(code_to_exec)
                analyzer = FindDependentKernelVariables(self.arg_names)
                analyzer.visit(tree)

                # analyzer.depends_on now has all kernel arg variables, 
                # used in the construction of the matrix

                if analyzer.depends_on:                 
                    # Need arg structure, python code, and locals[vars] code
                    arg_struct = ','.join([self.allowed_type_cpp_map[str(
                        self.type_annotations[s])]+' '+s for s in analyzer.depends_on])
                    arg_var_names = ','.join(
                        [s for s in analyzer.depends_on])
                    locals_code = '\n'.join(
                        ['locals["{}"] = {};'.format(n, n) for n in arg_var_names])
                    self.extra_cpp_code = cpp_matrix_gen_code.format(
                        with_decomp_matrix_names[i], arg_struct, code_to_exec, locals_code)

                    col_skip = ' '*with_decomp_lines_col_starts[i]
                    new_src = col_skip + 'decompose {\n'
                    new_src += col_skip + ' '*4 + \
                        'auto [mat_data, mat_size] = __internal__qcor_pyjit_gen_{}_unitary_matrix({});\n'.format(
                            with_decomp_matrix_names[i], arg_var_names)
                    new_src += col_skip+' '*4 + \
                        'UnitaryMatrix {} = Eigen::Map<UnitaryMatrix>(mat_data.data(), mat_size, mat_size);\n'.format(
                            with_decomp_matrix_names[i])
                    new_src += col_skip + \
                        '{}({});\n'.format('}', decompose_args)
                    fbody_src = fbody_src.replace(
                        total_decompose_code, new_src)
                else:
                    # Execute the code, extract the matrix data and size
                    # This is the case where the matrix is static and does 
                    # not depend on any kernel arguments
                    _locals = locals()
                    exec(code_to_exec, globals(), _locals)
                    data = _locals['mat_data']
                    data = ','.join([str(d) for d in data])
                    mat_size = _locals['mat_size']

                    # Replace total_decompose_code in fbody_src...
                    col_skip = ' '*with_decomp_lines_col_starts[i]
                    new_src = col_skip + 'decompose {\n'
                    new_src += col_skip+' '*4 + 'UnitaryMatrix {} = UnitaryMatrix::Zero({},{});\n'.format(
                        with_decomp_matrix_names[i], mat_size, mat_size)
                    new_src += col_skip+' '*4 + \
                        '{} << {};\n'.format(with_decomp_matrix_names[i], data)
                    new_src += col_skip + \
                        '{}({});\n'.format('}', decompose_args)
                    fbody_src = fbody_src.replace(
                        total_decompose_code, new_src)

        if 'with compute' in fbody_src:
            # All we really should need to do is 
            # convert with compute to compute { 
            # and with action to } action {
            # then close with a } when we 
            # hit a new col location
            
            assert(fbody_src.count('with compute') == fbody_src.count('with action'))

            # split the function into lines
            lines = fbody_src.split('\n')
            new_src = ''
            in_action_block = False
            in_compute_block = False
            action_col, compute_col = (0, 0)
            for line in lines:
                current_col = sum(1 for _ in itertools.takewhile(str.isspace, line)) 
                if in_action_block and current_col <= action_col:
                    new_src += '}\n'

                if in_compute_block and current_col <= compute_col:
                    # here we have just dropped out of compute col
                    if 'with action' not in line:
                        print('After compute block, you must provide the action block.')
                        exit(1)

                if 'with compute' in line:
                    in_action_block = False
                    in_compute_block = True
                    compute_col = sum(1 for _ in itertools.takewhile(str.isspace, line))+1
                    new_src += 'compute {\n'
                elif 'with action' in line:
                    in_action_block = True
                    in_compute_block = False
                    action_col = sum(1 for _ in itertools.takewhile(str.isspace, line)) 
                    new_src += '} action {\n'
                else:
                    new_src += line + '\n'
                
            # update the source code
            fbody_src = new_src

        # Users must provide arg types, if not we throw an error
        if not self.type_annotations or len(self.arg_names) != len(self.type_annotations):
            print('Error, you must provide type annotations for qcor quantum kernels.')
            exit(1)

        # Construct the C++ kernel arg string
        cpp_arg_str = ''
        self.ref_type_args = []
        self.qRegName = ''
        for arg, _type in self.type_annotations.items():
            if _type is FLOAT_REF:
                self.ref_type_args.append(arg)
                cpp_arg_str += ',' + \
                    'double& ' + arg
                continue
            if _type is INT_REF:
                self.ref_type_args.append(arg)
                cpp_arg_str += ',' + \
                    'int& ' + arg
                continue
            if str(_type).startswith('typing.Callable'):
                cpp_type_str = 'KernelSignature<'
                for i in range(len(_type.__args__) - 1):
                    # print("input type:", _type.__args__[i])
                    arg_type = _type.__args__[i]
                    if str(arg_type) not in self.allowed_type_cpp_map:
                        print('Error, this quantum kernel arg type is not allowed: ', str(_type))
                        exit(1)
                    cpp_type_str += self.allowed_type_cpp_map[str(arg_type)]
                    cpp_type_str += ','
                
                cpp_type_str = cpp_type_str[:-1]
                cpp_type_str += '>'
                # print("cpp type", cpp_type_str)
                cpp_arg_str += ',' + cpp_type_str + ' ' + arg
                continue

            if str(_type) not in self.allowed_type_cpp_map:
                print('Error, this quantum kernel arg type is not allowed: ', str(_type))
                exit(1)
            if self.allowed_type_cpp_map[str(_type)] == 'qreg':
                self.qRegName = arg
            cpp_arg_str += ',' + \
                self.allowed_type_cpp_map[str(_type)] + ' ' + arg
        cpp_arg_str = cpp_arg_str[1:]

        globalVarDecl = []
        # Get all globals currently defined at this stack frame
        globalsInStack = inspect.stack()[1][0].f_globals
        globalVars = globalsInStack.copy()
        importedModules = {}
        for key in globalVars:
            descStr = str(globalVars[key])
            # Cache module import and its potential alias
            # e.g. import abc as abc_alias
            if descStr.startswith("<module "):
                moduleName = descStr.split()[1].replace("'", "")
                importedModules[key] = moduleName
            elif key in fbody_src:
                # Import global variables (if used in the body):
                # Only support float atm
                if (isinstance(globalVars[key], float)):
                    globalVarDecl.append(key + " = " + str(globalVars[key]))

        # Inject these global declarations into the function body.
        separator = "\n"
        globalDeclStr = separator.join(globalVarDecl)

        # Handle common modules like numpy or math
        # e.g. if seeing `import numpy as np`, we'll have <'np' -> 'numpy'> in the importedModules dict.
        # We'll replace any module alias by its original name,
        # i.e. 'np.pi' -> 'numpy.pi', etc.
        for moduleAlias in importedModules:
            if moduleAlias != importedModules[moduleAlias]:
                aliasModuleStr = moduleAlias + '.'
                originalModuleStr = importedModules[moduleAlias] + '.'
                fbody_src = fbody_src.replace(
                    aliasModuleStr, originalModuleStr)

        # Persist *pass by ref* variables to the accelerator buffer:
        persist_by_ref_var_code = ''
        for ref_var in self.ref_type_args:
            persist_by_ref_var_code += '\npersist_var_to_qreq(\"' + \
                ref_var + '\", ' + ref_var + ', ' + self.qRegName + ')'

        # Create the qcor quantum kernel function src for QJIT and the Clang syntax handler
        self.src = '__qpu__ void '+self.function.__name__ + \
            '('+cpp_arg_str+') {\nusing qcor::pyxasm;\n' + \
            globalDeclStr + '\n' + fbody_src + persist_by_ref_var_code + "}\n"

        # Handle nested kernels:
        dependency = []
        for kernelName in self.__compiled__kernels:
            # Check that this kernel *calls* a previously-compiled kernel:
            # pattern: "<white space> kernel(" OR "kernel.adjoint(" OR "kernel.ctrl("
            kernelCall = kernelName + '('
            kernelAdjCall = kernelName + '.adjoint('
            kernelCtrlCall = kernelName + '.ctrl('
            if re.search(r"\b" + re.escape(kernelCall) + '|' + re.escape(kernelAdjCall) + '|' + re.escape(kernelCtrlCall), self.src):
                dependency.append(kernelName)

        self.__kernels__graph.addKernelDependency(
            self.function.__name__, dependency)
        self.sorted_kernel_dep = self.__kernels__graph.getSortedDependency(
            self.function.__name__)

        # print(self.src)
        # Run the QJIT compile step to store function pointers internally
        self._qjit.internal_python_jit_compile(
            self.src, self.sorted_kernel_dep, self.extra_cpp_code, extra_headers)
        self._qjit.write_cache()
        self.__compiled__kernels.append(self.function.__name__)
        QJIT_OBJ_CACHE.append(self)
        return

    # Static list of all kernels compiled
    __compiled__kernels = []
    __kernels__graph = KernelGraph()

    def get_syntax_handler_src(self):
        """
        Good for debugging purposes - return the actuall C++ code that the SyntaxHandler
        generates for this qjit kernel.
        """
        return self._qjit.run_syntax_handler(self.src)[1]

    def get_extra_cpp_code(self):
        """
        Return any required C++ code that the JIT source code will need.
        """
        return self.extra_cpp_code

    def get_sorted_kernels_deps(self):
        return self.sorted_kernel_dep

    def get_internal_src(self):
        """
        Return the C++ / embedded python DSL function code that will be passed to QJIT
        and the clang syntax handler. This function is primarily to be used for developer purposes.
        """
        return self.src

    def kernel_name(self):
        """Return the quantum kernel function name."""
        return self.function.__name__

    def translate(self, q: qreg, x: List[float]):
        """
        This method is primarily used internally to map Optimizer parameters x : List[float] to 
        the argument structure expected by the quantum kernel. For example, for a kernel 
        expecting (qreg, float) arguments, this method should return a dictionary where argument variable 
        names serve as keys, and values are corresponding argument instances. Specifically, the float 
        argument variable should point to x[0], for example. 
        """

        # Local vars used to figure out if we have
        # arg structures that look like (qreg, float...)
        type_annots_list = [str(self.type_annotations[x])
                            for x in self.arg_names]
        default_float_args = ['<class \'float\'>']
        intersection = list(
            set(type_annots_list[1:]) & set(default_float_args))

        if intersection == default_float_args:
            # This handles all (qreg, float...)
            ret_dict = {self.arg_names[0]: q}
            for i, arg_name in enumerate(self.arg_names[1:]):
                ret_dict[arg_name] = x[i]
            if len(ret_dict) != len(self.type_annotations):
                print(
                    'Error, could not translate vector parameters x into arguments for quantum kernel. ', len(ret_dict), len(self.type_annotations))
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
                print(
                    'Error, could not translate vector parameters x into arguments for quantum kernel.')
                exit(1)
            return ret_dict
        else:
            print('currently cannot translate other arg structures')
            exit(1)

    def extract_composite(self, *args):
        """
        Convert the quantum kernel into an XACC CompositeInstruction
        """
        assert len(args) == len(self.arg_names), "Cannot create CompositeInstruction, you did not provided the correct kernel arguments."
        # Create a dictionary for the function arguments
        args_dict = self.construct_arg_dict(*args)
        return self._qjit.extract_composite(self.function.__name__, args_dict)

    def observe(self, observable, *args):
        """
        Return the expectation value of <observable> with 
        respect to the state given by this qjit kernel evaluated 
        at the given arguments. 
        """
        program = self.extract_composite(*args)
        return internal_observe(program, observable)

    def openqasm(self, *args):
        """
        Return an OpenQasm string representation of this 
        quantum kernel.
        """
        kernel = self.extract_composite(*args)
        staq = xacc.getCompiler('staq')
        return staq.translate(kernel)

    def print_kernel(self, *args):
        """
        Print the QJIT kernel as a QASM-like string
        """
        print(self.extract_composite(*args).toString())

    def n_instructions(self, *args):
        """
        Return the number of quantum instructions in this kernel. 
        """
        return self.extract_composite(*args).nInstructions()
    
    def as_unitary_matrix(self, *args):
        args_dict = self.construct_arg_dict(*args)
        return self._qjit.internal_as_unitary(self.function.__name__, args_dict)
    
    def ctrl(self, *args):
        assert False, 'This is an internal API call and will be translated to C++ via the QJIT.\nIt can only be called from within another quantum kernel.'

    def adjoint(self, *args):
        assert False, 'This is an internal API call and will be translated to C++ via the QJIT.\nIt can only be called from within another quantum kernel.'

    def mlir(self, *args, **kwargs):
        assert len(args) == len(self.arg_names), "Cannot generate MLIR, you did not provided the correct concrete kernel arguments."
        open_qasm_str = self.openqasm(*args)
        return openqasm_to_mlir(open_qasm_str, self.kernel_name(), 
                        kwargs['add_entry_point'] if 'add_entry_point' in kwargs else True)
    
    def llvm_mlir(self, *args, **kwargs):
        assert len(args) == len(self.arg_names), "Cannot generate LLVM MLIR, you did not provided the correct concrete kernel arguments."
        open_qasm_str = self.openqasm(*args)
        return openqasm_to_llvm_mlir(open_qasm_str, self.kernel_name(), 
                        kwargs['add_entry_point'] if 'add_entry_point' in kwargs else True)

    def llvm_ir(self, *args, **kwargs):
        assert len(args) == len(self.arg_names), "Cannot generate LLVM IR, you did not provided the correct concrete kernel arguments."
        open_qasm_str = self.openqasm(*args)
        return openqasm_to_llvm_ir(open_qasm_str, self.kernel_name(), 
                        kwargs['add_entry_point'] if 'add_entry_point' in kwargs else True)

    def qir(self, *args, **kwargs):
        return llvm_ir(*args, **kwargs)

    # Helper to construct the arg_dict (HetMap)
    # e.g. perform any additional type conversion if required.
    def construct_arg_dict(self, *args):
        # Create a dictionary for the function arguments
        args_dict = {}
        for i, arg_name in enumerate(self.arg_names):
            args_dict[arg_name] = list(args)[i]
            arg_type_str = str(self.type_annotations[arg_name])
            if arg_type_str.startswith('typing.Callable'):
                # print("callable:", arg_name)
                # print("arg:", type(args_dict[arg_name]))
                # the arg must be a qjit
                if not isinstance(args_dict[arg_name], qjit):
                    print('Invalid argument type for {}. A quantum kernel (qjit) is expected.'.format(arg_name))
                    exit(1)
                
                callable_qjit = args_dict[arg_name]
                fn_ptr = self._qjit.get_kernel_function_ptr(callable_qjit.kernel_name())
                if fn_ptr == 0:
                    print('Failed to retrieve JIT-compiled function pointer for qjit kernel {}.'.format(callable_qjit.kernel_name()))
                    exit(1)
                # Replace the argument (in the dict) with the function pointer
                # qjit is a pure-Python object, hence cannot be used by native QCOR.
                args_dict[arg_name] = hex(fn_ptr)
            
        return args_dict

    def __call__(self, *args):
        """
        Execute the decorated quantum kernel. This will directly 
        invoke the corresponding LLVM JITed function pointer. 
        """
        arg_dict = self.construct_arg_dict(*args)
        # Invoke the JITed function
        self._qjit.invoke(self.function.__name__, arg_dict)

        # Update any *by-ref* arguments: annotated with the custom type: FLOAT_REF, INT_REF, etc.
        # If there are *pass-by-ref* variables:
        if len(self.ref_type_args) > 0:
            # Access the register:
            qReg = args_dict[self.qRegName]
            # Retrieve *original* variable names of the argument pack
            frame = inspect.currentframe()
            frame = inspect.getouterframes(frame)[1]
            code_context_string = inspect.getframeinfo(
                frame[0]).code_context[0].strip()
            caller_args = code_context_string[code_context_string.find(
                '(') + 1:-1].split(',')
            caller_var_names = []
            for i in caller_args:
                i = i.strip()
                if i.find('=') != -1:
                    caller_var_names.append(i.split('=')[1].strip())
                else:
                    caller_var_names.append(i)

            # Get the updated value:
            for by_ref_var in self.ref_type_args:
                updated_var = qReg.getInformation(by_ref_var)
                caller_var_name = caller_var_names[self.arg_names.index(
                    by_ref_var)]
                if (caller_var_name in inspect.stack()[1][0].f_globals):
                    # Make sure it is the correct type:
                    by_ref_instane = inspect.stack(
                    )[1][0].f_globals[caller_var_name]
                    # We only support float and int atm
                    if (isinstance(by_ref_instane, float) or isinstance(by_ref_instane, int)):
                        inspect.stack()[
                            1][0].f_globals[caller_var_name] = updated_var

        return

class KernelBuilder(object):
    """
    The QCOR KernelBuilder is a high-level data structure that enables the 
    development of qcor quantum kernels programmatically in Python. Example usage:

    from qcor import * 

    nq = 10
    builder = KernelBuilder() 

    builder.h(0)
    for i in range(nq-1):
        builder.cnot(i, i+1)
    builder.measure_all()
    ghz = builder.create()

    q = qalloc(nq)
    ghz(q)
    print(q.counts())

    If you do not provide a qreg argument to the constructor (py_args_dict) 
    we will assume a single qreg named q.
    """
    def __init__(self,**kwargs):
        self.kernel_args = kwargs['kernel_args'] if 'kernel_args' in kwargs else {}
        # Returns list of tuples, (name, nRequiredBits, isParameterized)
        all_instructions = internal_get_all_instructions()
        all_instructions = [element for element in all_instructions if element[0] != 'Measure']
        self.qjit_str = ''
        self.qreg_name = 'q'
        self.TAB = '    '

        for instruction in all_instructions:
            isParameterized = instruction[2]
            n_bits = instruction[1]
            name = instruction[0]

            qbits_str = ','.join(['q{}'.format(i) for i in range(n_bits)])
            qbits_indexed = ','.join(["{}[{{}}]".format(self.qreg_name) for i in range(n_bits)])
            new_func_str = '''def {}(self, {}, *args):
    params_str = ''
    params = []
    if len(args):
        for arg in args:
            if isinstance(arg, str):
                params.append(str(arg))
                if str(arg) not in self.kernel_args:
                    self.kernel_args[str(arg)] = float
            elif isinstance(arg, tuple):
                params.append(arg[0]+'['+str(arg[1])+']')
                if arg[0] not in self.kernel_args:
                    self.kernel_args[arg[0]] = List[float]
            else:
                print('[KernelBuilder Error] Invalid parameter type.')
                exit(1)
        params_str = ','.join(params)
    if {} and len(args) == 0:
        print("[KernelBuilder Error] You are calling a parameterized instruction ({}) but have not provided any parameters")
        exit(1)
    if not params_str:
        self.qjit_str += self.TAB+'{}({})\\n'.format({})
    else:
        self.qjit_str += self.TAB+'{}({}, {{}})\\n'.format({}, params_str)
'''.format(name.lower(), qbits_str, isParameterized, name.lower(), name, qbits_indexed, qbits_str, name, qbits_indexed, qbits_str)
            # print(new_func_str)
            result = globals()
            exec (new_func_str.strip(), result)
            setattr(KernelBuilder, instruction[0].lower(), result[instruction[0].lower()])

    def measure_all(self):
        self.qjit_str += '\n'
        self.qjit_str += self.TAB + 'for i in range({}.size()):\n'.format(self.qreg_name)
        self.qjit_str += self.TAB+self.TAB+'Measure({}[i])\n'.format(self.qreg_name)

    def measure(self, qbit): 
        if isinstance(qbit, range):
            qbit = list(qbit)
        if isinstance(qbit, list):
            for i in qbit:
                self.measure(i)
        elif isinstance(qbit, int):
            self.qjit_str += self.TAB+'Measure({}[{}])\n'.format(self.qreg_name, qbit)
        else:
            print('[KernelBuilder] invalid input to measure {}'.format(qbit))
            exit(1)

    def exp(self, variable, op : xacc.Observable):
        params_str = ''
        if isinstance(variable, str):
            params_str = variable
            if str(variable) not in self.kernel_args:
                self.kernel_args[str(variable)] = float
        elif isinstance(variable, tuple):
            params_str = variable[0]+'['+str(variable[1])+']'
            if variable[0] not in self.kernel_args:
                self.kernel_args[variable[0]] = List[float]
        else:
            print('[KernelBuilder Error] Invalid exp() parameter type.')
            exit(1)

        op_str = op.toString()
        op_type = 'fermion' if '^' in op_str else 'pauli'
        hash_object = hashlib.md5(op_str.encode('utf-8'))
        op_var_name = '__internal_op_var_'+str(hash_object.hexdigest())
        self.qjit_str += self.TAB+"{} = _internal_python_createObservable(\"{}\", \"{}\")\n".format(op_var_name, op_type, op_str)
        self.qjit_str += self.TAB+'exp_i_theta({}, {}, {})\n'.format(self.qreg_name, params_str, op_var_name)

    def invoke(self, kernel_function : qjit):
        arg_names, _, _, _, _, _, type_annotations = inspect.getfullargspec(
            kernel_function)
        args_dict = {k:v for k,v in kernel_function.type_annotations.items() if str(typing_to_simple_map[str(v)]) != 'qreg'}
        self.kernel_args = {**args_dict, **self.kernel_args}
        self.qjit_str+= self.TAB+'{}({}, {})\n'.format(kernel_function.kernel_name(), self.qreg_name, ','.join(list(self.kernel_args.keys())))

    def from_qasm(self, qasm_str):
        xacc_ir = xacc.getCompiler('staq').compile(qasm_str).getComposites()[0]
        pyxasm = xacc.getCompiler('pyxasm').translate(xacc_ir)
        pyxasm = pyxasm.replace('__translate_qrg__', self.qreg_name)
        pyxasm = '\n'.join(self.TAB+line for line in pyxasm.split('\n'))
        processed_str = pyxasm        
        self.qjit_str += processed_str

    def from_qiskit(self, qk_circ):
        return self.from_qasm(qk_circ.qasm())
    
    # Synthesis from matrix, or from matrix generator
    # can provide method = [qsearch,qfast,kak, etc.]
    def synthesize(self, **kwargs):
        method = ''
        if 'method' in kwargs:
            method = kwargs['method']

        if 'unitary' not in kwargs:
            print('[KernelBuilder Error] Please pass unitary=matrix_var kwarg.')
        
        unitary = kwargs['unitary']

        if hasattr(unitary, '__call__'):
            fbody_src = '\n'.join(inspect.getsource(unitary).split('\n')[1:])
            hash_object = hashlib.md5(fbody_src.encode('utf-8'))
            arg_names, _, _, _, _, _, type_annotations = inspect.getfullargspec(unitary)

            # add arguments if not in self.kernel_args
            for arg, t in type_annotations.items():
                if arg not in self.kernel_args:
                    # print('[KernelBuilder] Found argument in unitary generator that is unknown ({}). Adding it to the kernel args.'.format(arg))
                    self.kernel_args[arg] = t


            mat_var_name = '__internal_matrix_data_'+str(hash_object.hexdigest())
            self.qjit_str += self.TAB + 'with decompose(q{}) as {}:\n'.format(','+method if method!='' else method, mat_var_name)
            for line in fbody_src.split('\n'):
                line = ' '.join(line.split())
                if 'return' in line:
                    line = line.replace('return', mat_var_name+' =')
                self.qjit_str += self.TAB + self.TAB + line + '\n'

        elif hasattr(unitary, '__iter__'):
            unitary_str = ' ; '.join([str(row).replace('[','').replace(']','').replace(' ', ', ') for row in unitary])
            hash_object = hashlib.md5(unitary_str.encode('utf-8'))
            mat_var_name = '__internal_matrix_data_'+str(hash_object.hexdigest())
            self.qjit_str += self.TAB + 'with decompose(q) as {}:\n'.format(mat_var_name)
            self.qjit_str += self.TAB+self.TAB+'{} = np.matrix("{}")\n'.format(mat_var_name, unitary_str)
        else:
            print('[KernelBuilder Error] Cannot parse this type of unitary matrix')
            exit(1)

    def create(self):
        # print(self.qjit_str)

        kernel_name = '__internal_qjit_kernelbuilder_kernel_'+str(uuid.uuid4()).replace('-','_')
        for element in inspect.stack():
            if len(element.code_context) and element.code_context[0] != None:
                if '.create()' in element.code_context[0]:
                    kernel_name = element.code_context[0].strip().split(' = ')[0]
                    break

        # FIXME optionally add , if we have kernel_args
        args_str = 'q : qreg'+ (', ' if len(self.kernel_args) else '') + ', '.join(k+' : '+typing_to_simple_map[str(v)] for k,v in self.kernel_args.items())
        func = 'def {}({}):\n'.format(kernel_name, args_str)+self.qjit_str
        # print(func)
        result = globals()
        exec(func, result)
        # print(result)
        function = result[kernel_name]
        _qjit = qjit(function, __internal_fbody_src_provided__ = self.qjit_str)
        return _qjit
        

# Must have qpu in the init kwargs
# defaults to qpp, but will search for -qpu flag
init_kwargs = {'qpu': sys.argv[sys.argv.index(
    '-qpu')+1] if '-qpu' in sys.argv else 'qpp'}

# get shots if provided
if '-shots' in sys.argv:
    init_kwargs['shots'] = int(sys.argv[sys.argv.index('-shots')+1])

# get -qrt if provided
if '-qrt' in sys.argv:
    init_kwargs['qrt'] = sys.argv[sys.argv.index('-qrt')+1]

# get Pass Manager configs:
if '-opt' in sys.argv:
    init_kwargs['opt'] = int(sys.argv[sys.argv.index('-opt')+1])

if '-print-opt-stats' in sys.argv:
    init_kwargs['print-opt-stats'] = True

if '-placement' in sys.argv:
    init_kwargs['placement'] = sys.argv[sys.argv.index('-placement')+1]

if '-opt-pass' in sys.argv:
    init_kwargs['opt-pass'] = sys.argv[sys.argv.index('-opt-pass')+1]

if '-qubit-map' in sys.argv:
    init_kwargs['qubit-map'] = sys.argv[sys.argv.index('-qubit-map')+1]

# Implements internal_startup initialization:
# i.e. set up qrt, backends, shots, etc.
Initialize(**init_kwargs)
