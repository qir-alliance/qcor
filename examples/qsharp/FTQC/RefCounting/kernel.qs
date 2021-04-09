namespace QCOR 
{ 
operation TestClean() : Unit {
    // Arrays
    let test1 = [1.0, 2.0];
    let test2 = [1, 2, 3, 4];
    // Tuples
    let test4 = (1.0, [2, 3, 4]);
}

operation TestKernel() : Double[] {
    // Arrays
    let test1 = [1.0, 2.0];
    let test2 = [1, 2, 3, 4];
    // Tuples
    let test4 = (1.0, [2, 3, 4]);
    // Return an array
    return [5.0, 6.0, 7.0];
}
}