OPENQASM 3;
include "qelib1.inc";

int[32] i = 20;
int[32] j ;

while (j < i) {
    print(j);
    j += 1;
}

print("starting next loop");
while (j < 100) {
    j += 1;
    if (j == 33) {
        print("skipping 33 to show continue directive.");
        continue;
    }

    if (j == 98) {
        print("exiting loop at 98 to show break directive.");
        break;
    }
    print(j);
}
