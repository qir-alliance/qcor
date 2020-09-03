#include <qalloc>

__qpu__ void ansatz(qreg q, std::vector<double> params){
    Ry(q[0], params[0]);
    Rx(q[1], params[1]);
}

int main(int argc, char **argv){
    auto q = qalloc(2);

    auto n_variational_params = 2;
    auto objective = createObjectiveFunction("ddcl", ansatz, q, n_variational_params, 
    {{"target-dist", std::vector<double> {0.5, 0.0, 0.5, 0.0} }} );
    std::vector<double> initial_params = {0.0, 0.0};

    auto optimizer = createOptimizer("nlopt", std::make_pair("initial-parameters", initial_params));
    set_verbose(true);
    auto handle = taskInitiate(objective, optimizer);
    std::cout<<"task initiated!\n";

    auto results = sync(handle);
    //printf("results: %f\n", results.opt_val);
}