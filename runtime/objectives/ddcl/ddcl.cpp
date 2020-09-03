#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

#include "AlgorithmGradientStrategy.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

using namespace cppmicroservices;

namespace qcor{
class DDCLObjective : public ObjectiveFunction{
public:
    std::shared_ptr<xacc::Algorithm> ddcl;
    double operator()(xacc::internal_compiler::qreg &qreg,
                        std::vector<double> &dx) override {
        if(!ddcl){
            ddcl = xacc::getAlgorithm("ddcl");
        }
        std::vector<double> target_dist;
        //(!parameters.keyExists<std::vector<double>>("target_dist"))
        if(this->options.keyExists<std::vector<double>>("target-dist")){
            target_dist = this->options.get<std::vector<double>>("target-dist");
        }
        else{
            xacc::error("no target distrbution recieved!\n");
        }
        std::string loss;
        if(this->options.keyExists<std::string>("loss")){
            loss = this->options.get<std::string>("loss");
        }
        else{
            loss = "js";
        }
        for(auto &x:target_dist){
            std::cout<<x<<" ";
        }
        std::cout<<"\n";
        auto qpu = xacc::internal_compiler::get_qpu();
        auto success = ddcl->initialize(
            {{"ansatz", kernel}, {"accelerator", qpu}, {"target-dist", target_dist},
            {"loss", loss}});
        if(!success){
            xacc::error("QCOR DDCL Error - could not initialize internal xacc ddcl algorithm.");
        }
        auto tmp_child = qalloc(qreg.size());
        auto val = ddcl->execute(xacc::as_shared_ptr(tmp_child.results()), {})[0];
        for(auto &child : tmp_child.results()->getChildren()){
            child->addExtraInfo("parameters", current_iterate_parameters);
            auto tmp = current_iterate_parameters;
            tmp.push_back(val);
            child->addExtraInfo("qcor-params-loss", tmp);
        }
        qreg.addChild(tmp_child);
    return val;
    }
    public:
        const std::string name() const override {return "ddcl";}
        const std::string description() const override {return "";}
};
}//namespace qcor

namespace {
    class US_ABI_LOCAL DDCLObjectiveActivator : public BundleActivator {
    public:
        DDCLObjectiveActivator() {}

        void Start(BundleContext context){
            auto xt = std::make_shared<qcor::DDCLObjective>();
            context.RegisterService<qcor::ObjectiveFunction>(xt);
        }

        void Stop(BundleContext) {}
    };
} //namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(DDCLObjectiveActivator)
