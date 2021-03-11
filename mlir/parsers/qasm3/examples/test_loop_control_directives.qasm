OPENQASM 3;
include "stdgates.inc";

for i in [0:10] {
    if (i == 5) {
        print("breaking at 5");
        break;
    }
    if (i == 2) {
        print("continuing at 2");
        continue;
    }
    print("i = ", i);
}

print("made it out of the loop");