add_mlir_dialect_library(MLIRQuantum
        QuantumDialect.cpp
        QuantumOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Quantum

        DEPENDS
        MLIRQuantumOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
