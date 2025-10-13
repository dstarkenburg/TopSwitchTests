using PGLib
using PowerModels
using JuMP, Ipopt
using Plots
using Gurobi
using Zygote, LinearAlgebra, SparseArrays
using HDF5

using HSL
import HSL_jll
bool = LIBHSL_isfunctional()
@info "HSL solvers are working: "*string(bool)

# see here: https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#pythoncall-config
    # => these env definitions must go BEFORE "using PythonCall"
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/home/schev/pyverify/bin/python3"

using PythonCall
using MathOptAI

sys = pyimport("sys")
println("Python version: ", sys.version)
println("Virtual Env Location: ", sys.prefix)

# ================
include("./functions.jl")
include("./structs.jl")

# %% call the test case
include("./functions.jl")

nl_list   = [20; 38; 186]
bus_list  = [14; 24; 118]
node_list = [32; 128; 512; 2048]
pg_list   = ["case14_"; "case24_ieee"; "pglib_opf_case118_ieee.m"]

#  bounds over which to verify
bounds = Dict(:load_scale_lb => 0.75,   :load_scale_ub  => 1.25,
              :risk_lb       => 0.25,   :risk_ub        => 0.75,
              :alpha_lb      => 0.25,   :alpha_ub       => 0.75)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
include("./functions.jl")

ii = 3
node = node_list[4]
bus = bus_list[ii]
network_data = pglib(pg_list[ii])
gm = parse_PM_to_SOCGridModel(network_data; perturb=false)


# %% look at the 118 bus system
# step 1: sample from the bounds
pd0 = copy(gm.pd)
qd0 = copy(gm.qd)

p_lb, p_ub = scale_load(pd0, bounds[:load_scale_lb], bounds[:load_scale_ub])
q_lb, q_ub = scale_load(qd0, bounds[:load_scale_lb], bounds[:load_scale_ub])

bounds = Dict(:load_scale_lb => 0.75,  :load_scale_ub  => 1.25,
            :risk_lb       => 0.4,   :risk_ub          => 0.5,
            :alpha_lb      => 0.4,   :alpha_ub         => 0.5)

p_rand     = rand(gm.nb)
q_rand     = rand(gm.nb)
risk_rand  = rand(gm.nl)
alpha_rand = rand()

# instantiate
pd_sample    = p_rand.*p_lb                  .+ (1.0 .- p_rand).*p_ub
qd_sample    = q_rand.*q_lb                  .+ (1.0 .- q_rand).*q_ub
risk_sample  = risk_rand.*bounds[:risk_lb]   .+ (1.0 .- risk_rand).*bounds[:risk_ub]
alpha_sample = alpha_rand.*bounds[:alpha_lb] .+ (1.0 .- alpha_rand).*bounds[:alpha_ub]
x_input      = [risk_sample; qd_sample; pd_sample; alpha_sample]

# step 2: compute the line statuses
nn_model = "outputs/"*string(bus)*"_bus/"*string(bus)*"_bus_"*string(node)*"node.pt"

zl0, logit_zl0 = line_status(gm, bounds, nn_model; x_input=x_input)

#zl0, logit_zl0 = line_status(gm, bounds, nn_model; high_load=true)
gm_shed        = deepcopy(gm)
gm_shed.pd    .= bounds[:load_scale_ub]*copy(gm.pd)
gm_shed.qd    .= bounds[:load_scale_ub]*copy(gm.qd)

zl0[ zl0 .< 0.5]  .= 0.0
zl0[ zl0 .>= 0.5] .= 1.0

dual_soln      = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl0)
ss(dual_soln)


# %%
include("./functions.jl")

n_samples = 100
data_file = "data/"*string(gm.nb)*"bus_"*string(node)*"node_sampling_test.h5"
nn_model = "outputs/"*string(bus)*"_bus/"*string(bus)*"_bus_"*string(node)*"node.pt"

obj_soc_zrelax, obj_soc_zsnap, obj_acopf_zsnap, n_fails = loop_and_sample(gm, bounds, nn_model, data_file; n_samples = n_samples)

# %%
node = node_list[4]

include("./functions.jl")
nn_model = "outputs/"*string(bus)*"_bus/"*string(bus)*"_bus_"*string(node)*"node.pt"

maxmin_model = maxmin_loadshed(gm, bounds, nn_model; tol = 5e-3, hot_start=true, tmax = 3600.0, flowcuts=true);

# %%



























using PGLib
using PowerModels
using JuMP, Ipopt
using Plots
using Gurobi
using Zygote, LinearAlgebra, SparseArrays
using HDF5

using HSL
import HSL_jll
bool = LIBHSL_isfunctional()

# see here: https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#pythoncall-config
    # => these env definitions must go BEFORE "using PythonCall"
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/home/schev/pyverify/bin/python3"

using PythonCall
using MathOptAI

sys = pyimport("sys")
println("Python version: ", sys.version)
println("Virtual Env Location: ", sys.prefix)

# ================
include("./functions.jl")
include("./structs.jl")


# %% Test maxmin 
include("./functions.jl")
include("./structs.jl")

# all the test case
bus_list  = [14; 24; 118]
node_list = [32; 128; 512; 2048]
pg_list   = ["case14_"; "case24_ieee"; "pglib_opf_case118_ieee.m"]

# bounds over which to verify
bounds = Dict(:load_scale_lb => 0.75,   :load_scale_ub  => 1.25,
              :risk_lb       => 0.25,   :risk_ub        => 0.75,
              :alpha_lb      => 0.25,   :alpha_ub       => 0.75)


ii  = 1
bus = bus_list[ii]
network_data = pglib(pg_list[ii])
gm = parse_PM_to_SOCGridModel(network_data; perturb=false)
zl = ones(gm.nl)


mm = ac_redispatch_snapped(gm, zl)

# %% warmup: test the model reformulations
include("./functions.jl")

node_list = [32; 128; 512; 2048]
node = node_list[4]
m1 = min_loadshed_soc_primal_explicit(gm, zl);
m2 = min_loadshed_soc_primal(gm, zl);
m3 = min_loadshed_soc_dual(gm, zl; solver=:Gurobi);

# %% ===========
include("./functions.jl")
zl = zeros(gm.nl)
zl[1:2:end] .= 1.0

m1 = min_loadshed_soc_primal_explicit(gm, zl; soc=false);
m1_flowcuts = min_loadshed_soc_primal_explicit_flowcuts(gm, zl; soc=false);
m2_flowcuts = min_loadshed_soc_primal(gm, zl; normalize_shed=false, flow_cuts=true);
m3_flowcuts = min_loadshed_soc_dual(gm, zl; solver=:Gurobi, flow_cuts=true);

# %% loop over NN architectures
include("./functions.jl")
node = node_list[3]

nn_model = "outputs/"*string(bus)*"_bus/"*string(bus)*"_bus_"*string(node)*"node.pt"
maxmin_model = maxmin_loadshed(gm, bounds, nn_model; tol = 1e-3, hot_start=true, tmax = 30.0, flowcuts=true);

# %%
include("./functions.jl")
data_file = string(gm.nb)*"bus_"*string(node)*"node_MathOptAI_test.h5"
m_soc_zrelax, m_acopf_zrelax, m_soc_zsnap, m_acopf_zsnap = process_maxmin_loadshed(gm, bounds, nn_model, data_file, maxmin_model)


# %%
include("./functions.jl")
zl          = copy(zl0)
zl[ zl .<= 1e-6] .= 0.0 # for numerical stability

zl[ zl .< 0.3]  .= 0.0
zl[ zl .>= 0.3] .= 1.0

m_soc_zsnap      = min_loadshed_soc_primal_explicit(gm_shed, zl, normalize_shed=false)

# %% ========
zl0, logit_zl0 = line_status(gm, bounds, nn_model)
dual_soln      = min_loadshed_soc_dual(gm, zl0)

# %%
    x_input = value.(maxmin_model[:x])
    zl0, logit_zl0 = line_status(gm, bounds, nn_model; x_input=x_input)
    pd = value.(maxmin_model[:pd0_var])
    qd = value.(maxmin_model[:qd0_var])

gm_shed     = deepcopy(gm)
    gm_shed.pd .= copy(pd)
    gm_shed.qd .= copy(qd)
    zl          = copy(zl0)
    zl[ zl .<= 1e-6] .= 0.0 # for numerical stability

m_acopf_zrelax   = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, normalize_shed=false)

# %%
include("./functions.jl")

bus = bus_list[1]
network_data = pglib(pg_list[1])
node = node_list[1]

gm = parse_PM_to_SOCGridModel(network_data; perturb=false)
# %%
include("./functions.jl")

n_samples = 10
data_file = "data/"*string(gm.nb)*"bus_"*string(node)*"node_sampling_test.h5"
_,_,_,_, n_fails = loop_and_sample(gm, bounds, nn_model, data_file; n_samples = n_samples)

# %% =========

gm_shed     = deepcopy(gm)
#gm_shed.pd .= copy(pd0)
#gm_shed.qd .= copy(qd0)
#zl          = copy(zl0)
#zl[ zl .<= 1e-6] .= 0.0 # for numerical stability
zl = rand(gm.nl)

m_soc_zrelax     = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; normalize_shed=false)
solution_valid_1 = ss(m_soc_zrelax)   # test for optimization failure

# => m_acopf_zrelax   = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, host_start=true, host_start_model=m_soc_zrelax, normalize_shed=false)
m_acopf_zrelax   = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, normalize_shed=false)
solution_valid_2 = ss(m_acopf_zrelax) # test for optimization failure

# %%
zl[ zl .< 0.5]  .= 0.0
zl[ zl .>= 0.5] .= 1.0
m_soc_zsnap      = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl, normalize_shed=false)
solution_valid_3 = ss(m_soc_zsnap)    # test for optimization failure

# %% ======
# => m_acopf_zsnap    = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, host_start=true, host_start_model=m_soc_zsnap, normalize_shed=false)
m_acopf_zsnap    = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, host_start=true, host_start_model=m_soc_zsnap, normalize_shed=false)
solution_valid_4 = ss(m_acopf_zsnap)  # test for optimization failure



# %%

m_soc_zrelax     = min_loadshed_soc_primal_explicit_flowcuts(gm, zl; normalize_shed=false)
solution_valid_1 = ss(m_soc_zrelax)   # test for optimization failure


# => m_acopf_zrelax   = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, host_start=true, host_start_model=m_soc_zrelax, normalize_shed=false)
m_acopf_zrelax   = min_loadshed_soc_primal_explicit_flowcuts(gm, zl; soc=false, host_start=true, host_start_model=m_soc_zrelax, normalize_shed=false)
solution_valid_2 = ss(m_acopf_zrelax) # test for optimization failure


zl[ zl .< 0.5]  .= 0.0
zl[ zl .>= 0.5] .= 1.0
m_soc_zsnap      = min_loadshed_soc_primal_explicit_flowcuts(gm, zl, normalize_shed=false)
solution_valid_3 = ss(m_soc_zsnap)    # test for optimization failure


# => m_acopf_zsnap    = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl; soc=false, host_start=true, host_start_model=m_soc_zsnap, normalize_shed=false)
m_acopf_zsnap    = min_loadshed_soc_primal_explicit_flowcuts(gm, zl; soc=false, host_start=true, host_start_model=m_soc_zsnap, normalize_shed=false)
solution_valid_4 = ss(m_acopf_zsnap)  # test for optimization failure
