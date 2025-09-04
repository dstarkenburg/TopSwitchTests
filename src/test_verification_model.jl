# %% ============
using JuMP
using Gurobi
using Ipopt
using PGLib
using PowerModels
import InfrastructureModels
using Printf

# added:
# 1) set generator lower bounds to 0
# 2) parameterize loads and shunts with relaxed binaries
# 3) parameterize lines with relaxed binaries
# 4) set the objective to minimize the total amount of load shed
data = pglib("case5_")
data = pglib("case24_ieee")
#data = pglib("case14_")

PowerModels.calc_thermal_limits!(data)
PowerModels.standardize_cost_terms!(data, order=2)
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

# %% build the model
model = Model(JuMP.optimizer_with_attributes(Ipopt.Optimizer))#, "Presolve"=> 2,))

# create a simple classifier which maps load to line probablility
sigmoid(x) = @. 1/(1+exp(-x))
n_input  = 2*length(ref[:load]) + length(ref[:branch]) + 1
n_output = length(ref[:branch])
A = randn(n_output,n_input)
b = randn(n_output)
NN(x,A,b) = sigmoid(A*x+b)
x_input = randn(n_input)

# set gen lower bounds
@warn("Set the generation lower bounds to 0")
for i in keys(ref[:gen])
    println(ref[:gen][i]["pmin"])
    ref[:gen][i]["pmin"] = 0.0
end

# add line binaries 
@warn("adding line binaries")
@variable(model, 0.0 <= z_branch[l in keys(ref[:branch])] <= 1.0)

#TODO add a constraint which links the loads and NN inputs
#pd_nn = ref[:load][d]["pd"]^2 * z_demand[d] 

num_loads = length(keys(ref[:load]))
@variable(model, pd_variable[1:num_loads])
@variable(model, qd_variable[1:num_loads])
@variable(model, risk[1:20])
@variable(model, alpha[1])
#for d in keys(ref[:load])
#    pdi = ref[:load][d]["pd"]
#    qdi = ref[:load][d]["qd"]
#end

# TODO make sure we add bounds on the following inputs!
#x = [pd_variable;
#     qd_variable;
#     risk;     # these are variables [0.2 - 0.85]
#     alpha]    # this is a variable [0.2 - 0.85]

# probs = NN(x_in,A,b)

# something like this:
# probs, _ = MathOptAI.add_predictor(model, predictor, x)

#TODO add a NN linking constraint like this:
#probs = NN(x_input,A,b) # replace with torch model
#TODO make sure the NN and the line indices are the same
@constraint(model, [l in keys(ref[:branch])], z_branch[l] == 1.0)


# add shunt binaries 
@warn("adding shunt binaries")
@variable(model, 0.0 <= z_shunt[s in keys(ref[:shunt])] <= 1.0)

# add load binaries 
@warn("adding load binaries")
@variable(model, 0.0 <= z_demand[d in keys(ref[:load])] <= 1.0)

# gen variables
@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

# branch flow variables
@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])

# bus voltage variables
@variable(model, ref[:bus][i]["vmin"]^2 <= w[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"]^2)

@variable(model, wr[(i,j) in keys(ref[:buspairs])] )
@variable(model, wi[(i,j) in keys(ref[:buspairs])] )

# lanl-ansi/PowerModels.jl/src/core/ref.jl #117
wr_min, wr_max, wi_min, wi_max = ref_calc_voltage_product_bounds(ref[:buspairs])
for bp in keys(ref[:buspairs])
    JuMP.set_lower_bound(wr[bp], wr_min[bp])
    JuMP.set_upper_bound(wr[bp], wr_max[bp])
    JuMP.set_lower_bound(wi[bp], wi_min[bp])
    JuMP.set_upper_bound(wi[bp], wi_max[bp])
end

# Constraints

# Voltage Model
# relaxation_complex_product  https://github.dev/lanl-ansi/PowerModels.jl #17
for (i,j) in keys(ref[:buspairs])
    JuMP.@constraint(model, wr[(i,j)]^2 + wi[(i,j)]^2 <= w[i]*w[j])
end

# Nodal Constraints
@warn("adding load shed to power balance")
for (i,bus) in ref[:bus]
    # Build a list of the loads and shunt elements connected to the bus i
    # bus = ref(pm, nw, :bus, i)
    bus_arcs = ref[:bus_arcs][i]
    # bus_arcs_dc = ref(pm, nw, :bus_arcs_dc, i)
    # bus_arcs_sw = ref(pm, nw, :bus_arcs_sw, i)
    bus_gens = ref[:bus_gens][i]
    bus_loads = ref[:bus_loads][i]
    bus_shunts = ref[:bus_shunts][i]
    # bus_storage = ref(pm, nw, :bus_storage, i)

    bus_pd = Dict(k => ref[:load][k]["pd"] for k in bus_loads)
    bus_qd = Dict(k => ref[:load][k]["qd"] for k in bus_loads)

    bus_gs = Dict(k => ref[:shunt][k]["gs"] for k in bus_shunts)
    bus_bs = Dict(k => ref[:shunt][k]["bs"] for k in bus_shunts)

    JuMP.@constraint(model,
        sum(p[a] for a in bus_arcs)
        ==
        sum(pg[g] for g in bus_gens) 
        - sum(pd*z_demand[j] for (j,pd) in bus_pd)   
        - sum(gs*z_shunt[k] for (k,gs) in bus_gs)*w[i]
    )
    JuMP.@constraint(model,
        sum(q[a] for a in bus_arcs)
        ==
        sum(qg[g] for g in bus_gens)
        - sum(qd*z_demand[j] for (j,qd) in bus_qd)
        + sum(bs*z_shunt[k] for (k,bs) in bus_bs)*w[i]
    )
end


# Branch Constraints
@warn("adding line outage to the line flows")
@warn("adding big-M relaxation to the angle limits")
for (i,branch) in ref[:branch]
    # branch = ref(pm, nw, :branch, i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    f_idx = (i, f_bus, t_bus)
    t_idx = (i, t_bus, f_bus)

    g, b = calc_branch_y(branch)
    tr, ti = calc_branch_t(branch)
    g_fr = branch["g_fr"]
    b_fr = branch["b_fr"]
    g_to = branch["g_to"]
    b_to = branch["b_to"]
    tm = branch["tap"]

    p_fr  = p[f_idx]
    q_fr  = q[f_idx]
    w_fr  = w[f_bus]
    wr_ij = wr[(f_bus, t_bus)]
    wi_ij = wi[(f_bus, t_bus)]

    p_to = p[t_idx]
    q_to = q[t_idx]
    w_to = w[t_bus]

    # power flow
    JuMP.@constraint(model, p_fr == z_branch[i]*( (g+g_fr)/tm^2*w_fr + (-g*tr+b*ti)/tm^2*wr_ij + (-b*tr-g*ti)/tm^2*wi_ij) )
    JuMP.@constraint(model, q_fr == z_branch[i]*(-(b+b_fr)/tm^2*w_fr - (-b*tr-g*ti)/tm^2*wr_ij + (-g*tr+b*ti)/tm^2*wi_ij) )

    JuMP.@constraint(model, p_to == z_branch[i]*( (g+g_to)*w_to + (-g*tr-b*ti)/tm^2*wr_ij + (-b*tr+g*ti)/tm^2*-wi_ij) )
    JuMP.@constraint(model, q_to == z_branch[i]*(-(b+b_to)*w_to - (-b*tr+g*ti)/tm^2*wr_ij + (-g*tr-b*ti)/tm^2*-wi_ij) )

    # Angle Limit
    buspair = ref[:buspairs][(f_bus,t_bus)]
    if buspair["branch"]==i
        angmax = buspair["angmax"]
        angmin = buspair["angmin"]

        M_angle = 5.0
        JuMP.@constraint(model, wi_ij <= tan(angmax)*wr_ij + M_angle*z_branch[i])
        JuMP.@constraint(model, wi_ij >= tan(angmin)*wr_ij - M_angle*z_branch[i])

        # add angle cuts
        #wf_lb, wf_ub = InfrastructureModels.variable_domain(w_fr)
        #wt_lb, wt_ub = InfrastructureModels.variable_domain(w_to)
#
        #vf_lb, vf_ub = sqrt(wf_lb), sqrt(wf_ub)
        #vt_lb, vt_ub = sqrt(wt_lb), sqrt(wt_ub)
        #td_ub = angmax
        #td_lb = angmin
#
        #phi = (td_ub + td_lb)/2
        #d   = (td_ub - td_lb)/2
#
        #sf = vf_lb + vf_ub
        #st = vt_lb + vt_ub
#
        #JuMP.@constraint(model, sf*st*(cos(phi)*wr_ij + sin(phi)*wi_ij) - vt_ub*cos(d)*st*w_fr - vf_ub*cos(d)*sf*w_to >=  vf_ub*vt_ub*cos(d)*(vf_lb*vt_lb - vf_ub*vt_ub))
        #JuMP.@constraint(model, sf*st*(cos(phi)*wr_ij + sin(phi)*wi_ij) - vt_lb*cos(d)*st*w_fr - vf_lb*cos(d)*sf*w_to >= -vf_lb*vt_lb*cos(d)*(vf_lb*vt_lb - vf_ub*vt_ub))
    end

    # power limit
    rate_a = branch["rate_a"]
    JuMP.@constraint(model, p_fr^2 + q_fr^2 <= rate_a^2)
    JuMP.@constraint(model, p_to^2 + q_to^2 <= rate_a^2)
end

# maximize load served (i.e., minimize load shed) 0 < z <1
@objective(model, Max, sum(ref[:load][d]["pd"]^2 * z_demand[d] for d in keys(ref[:load])) + 
                       sum(ref[:shunt][s]["gs"]^2 * z_shunt[s] for s in keys(ref[:shunt])))

# Solve
optimize!(model)
objective_value(model)  |> println

# %% Parameterize the problem

