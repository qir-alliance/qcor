# MLIR as an IR enabling Quantum Optimizations
Here we demonstrate the utility of the MLIR for quantum compiler optimization passes. The specific goal is to highlight the need for an SSA-based IR for quantum computing. We want to demonstrate select common quantum optimizations and how the MLIR enables their implementation. 

## Goals

- Demonstrate the benefits of an SSA-based IR for quantum computing.
- Demonstrate common quantum optimizations (ID pairs, rotation merging, single-qubit gate merge, inlining, permute and cancel)
- Demonstrate classical optimizations that enable new quantum optimizations (inlining and loop unrolling)

## Outline
1. Start with `simple_opts.qasm`, walk through the various sections showing different code patterns that could be optimized (removed fully, or just reduced). Note what should remain after full optimization
2. Emit the unoptimized MLIR, note that all the quantum instructions are there (when they really do nothing).
3. Show the optimized MLIR, note how it was able to reduce to a couple instructions. 
4. Show the unoptimized LLVM/QIR
5. Show the optimized LLVM/QIR. 
6. Use `--pass-timing` to show exactly what was run.  
7. Move on to `trotter_unroll_simple.qasm` and note the `for` loop will naively place ID pairs next to each other. Show the unoptimized MLIR, noting the `affine.for` loop. 
8. Show the optimized MLIR, noting the `affine.for` is gone, and the compiler has optimized the code to the `for` loop body. 
9. Show the same for unoptimized and optimized LLVM. 
10. Open `trotter_unroll_with_inlining.qasm` to demonstrate a bit more complexity, and how inlining+ loop unrolling can work together to further optimized the code. Highlight the necessity of any Quantum IR to support both classical and quantum optimizations. 
11. Go through the same workflow, unoptimized/optimized MLIR, unoptimized/optimized QIR.
12. Show `-print-final-submission` for unoptimized and optimized.
## Notes:
