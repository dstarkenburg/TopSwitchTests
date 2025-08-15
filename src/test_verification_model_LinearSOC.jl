using JuMP
using Ipopt
using PGLib
using LinearSOCVerification

# %% ================
using PowerModels
_PM  = PowerModels
data = pglib("case5_")
pmv = _PM.instantiate_model(data, SOCWRPowerModel, LinearSOCVerification._build_relaxed_minimal_load_shed; ref_extensions=[_PM.ref_add_on_off_va_bounds!])
pmv.model

# solve
set_optimizer(pmv.model, Ipopt.Optimizer)
optimize!(pmv.model)
objective_value(pmv.model)
soln = LinearSOCVerification.collect_solution(_PM.ref, pmv.model)

# %% ===========
using PowerModels
using PowerModelsWildfire
using PowerModelsRestoration
using InfrastructureModels

_PM = PowerModels
_IM = InfrastructureModels
_PMW = PowerModelsWildfire
_PMR = PowerModelsRestoration

data = pglib("case5_")
data = pglib("case24_ieee")
pmr = _PM.instantiate_model(data, SOCWRPowerModel, LinearSOC._build_relaxed_minimal_load_shed; ref_extensions=[_PM.ref_add_on_off_va_bounds!])
pmr.model