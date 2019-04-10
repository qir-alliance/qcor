include_directories(${CMAKE_SOURCE_DIR}/ir/observable/chemistry)

add_xacc_test(ChemistryObservable)
target_link_libraries(ChemistryObservableTester qcor-observable-chemistry qcor)