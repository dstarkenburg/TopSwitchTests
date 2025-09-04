using PGLib
using PowerModels
using JuMP, Ipopt
using Plots
using Zygote, LinearAlgebra, SparseArrays

include("./functions.jl")

# call the test case
data = pglib("pglib_opf_case118_ieee.m")
data = pglib("case24_ieee")

# test ===
vm, va, pg, qg, Eg, Efr, Eto, g, g_fr, g_to, b, b_fr, b_to, tm, ta, pd, qd, gs, bs, nb, nl, ng, 
vmax, vmin, pg_max, pg_min, qg_max, qg_min, smax, clin, fr_buses, to_buses = sovle_acopf_directly(data, false)

# %% prepare the SOC formulation (note: scale ta by tm to be consistent with PM)
xtr = tm .* cos.(ta)
xti = tm .* sin.(ta)

# sparsify
Efr  = sparse(Efr)
Eto  = sparse(Eto)

# build the power flow matrices
Tpfr  = diagm(@. (g+g_fr)/tm^2)*Efr
TpRfr = diagm(@. (-g*xtr+b*xti)/tm^2)
TpIfr = diagm(@. (-b*xtr-g*xti)/tm^2)
Tqfr  = diagm(@. (-(b+b_fr)/tm^2))*Efr
TqRfr = diagm(@. -(-b*xtr-g*xti)/tm^2)
TqIfr = diagm(@. (-g*xtr+b*xti)/tm^2)
Tpto  = diagm(@. (g+g_to))*Eto
TpRto = diagm(@. (-g*xtr-b*xti)/tm^2)
TpIto = diagm(@. -(-b*xtr+g*xti)/tm^2)
Tqto  = diagm(@. -(b+b_to))*Eto
TqRto = diagm(@. -(-b*xtr+g*xti)/tm^2)
TqIto = diagm(@. -(-g*xtr-b*xti)/tm^2)

# loads
pd0 = copy(pd)
qd0 = copy(qd)

# grab the SOC bounds
wr_min, wr_max, wi_min, wi_max = soc_voltage_bound_vectors(data, nl, fr_buses, to_buses)

# shunt information
Bs_neg = diagm(min.(bs,0.0))
Bs_pos = diagm(max.(bs,0.0))
Gs     = diagm(gs)

# %% build soc_relaxation
model = Model(Ipopt.Optimizer)
zl = ones(nl)

# add lifted voltages
@variable(model, w[1:nb])
@variable(model, wr[1:nl])
@variable(model, wi[1:nl])
@variable(model, zd[1:nb])
@variable(model, pg[1:ng])
@variable(model, qg[1:ng])
@variable(model, pgs[1:nb])
@variable(model, qbs_pos[1:nb])
@variable(model, qbs_neg[1:nb])
@variable(model, p_fr[1:nl])
@variable(model, p_to[1:nl])
@variable(model, q_fr[1:nl])
@variable(model, q_to[1:nl])

# ================================================ #
# define the equality constraints
@constraint(model, p_fr .- zl.*(Tpfr*w + TpRfr*wr + TpIfr*wi) .== 0.0)
@constraint(model, p_to .- zl.*(Tpto*w + TpRto*wr + TpIto*wi) .== 0.0)
@constraint(model, q_fr .- zl.*(Tqfr*w + TqRfr*wr + TqIfr*wi) .== 0.0)
@constraint(model, q_to .- zl.*(Tqto*w + TqRto*wr + TqIto*wi) .== 0.0)

@constraint(model,  -Eg*pg + diagm(pd0)*zd + Efr'*p_fr + Eto'*p_to + pgs               .== 0.0)
@constraint(model,  -Eg*qg + diagm(qd0)*zd + Efr'*q_fr + Eto'*q_to - qbs_pos - qbs_neg .== 0.0)

# ================================================ #
# Define the inequality constraints
@constraint(model, 0.0 .- zd .<= 0.0)
@constraint(model, zd .- 1.0 .<= 0.0)

@constraint(model, vmin.^2 .- w .<= 0.0)
@constraint(model, w .- vmax.^2 .<= 0.0)

@constraint(model, wr_min .- wr .<= 0.0)
@constraint(model, wr .- wr_max .<= 0.0)

@constraint(model, wi_min .- wi .<= 0.0)
@constraint(model, wi .- wi_max .<= 0.0)

@constraint(model, 0.0 .- pg    .<= 0.0)
@constraint(model, pg .- pg_max .<= 0.0)

@constraint(model, qg_min .- qg .<= 0.0)
@constraint(model, qg .- qg_max .<= 0.0)

@constraint(model, -pgs               .<= 0.0)
@constraint(model, pgs - Gs*w         .<= 0.0)
@constraint(model, -qbs_pos           .<= 0.0)
@constraint(model, qbs_pos - Bs_pos*w .<= 0.0)
@constraint(model, Bs_neg*w - qbs_neg .<= 0.0)
@constraint(model, qbs_neg            .<= 0.0)

# ================================================ #
# Define the RSOC constraints
@constraint(model, p_fr.^2 .+ q_fr.^2 .<= smax.^2)
@constraint(model, p_to.^2 .+ q_to.^2 .<= smax.^2)

@constraint(model, (wr).^2 .+ (wi).^2 .<= (Efr*w).*(Eto*w))
# ================================================ #
# objective!
d1 = diagm(pd0)*zd - pd0
d2 = pgs -  Gs*w
d3 = qbs_pos - Bs_pos*w
d4 = Bs_neg*w - qbs_neg
@objective(model, Min, d1'*d1 + d2'*d2 + d3'*d3 + d4'*d4 + clin'*pg)
optimize!(model)
println(objective_value(model))

# %% canonicalize
# Ax + b  = 0
# Cx + d <= 0
# s in K
# => x = [w; wr; wi; zd; pg; qg; pgs; qbs_pos; qbs_neg; p_fr; p_to; p_fr; q_to]
include("./functions.jl")

clin = copy(clin0)
clin .= 0*10000.0
zl = 0.0
A, b, C, d, h, m1, b1, m2, b2, m3, b3 = canonicalize(zl, pd0, qd0, nb, nl, ng, Tpfr, TpRfr, TpIfr, Tpto, TpRto, TpIto, Tqfr, TqRfr, TqIfr, Tqto, TqRto, TqIto, Eg, Efr, Eto, vmin, vmax, wr_min, wr_max, wi_min, wi_max, pg_max, qg_min, qg_max, Gs, Bs_pos, Bs_neg, smax, clin) 

# solve the model
model = Model(Gurobi.Optimizer)
@variable(model, x[1:nvar])
@constraint(model, A*x + b .== 0.0)
@constraint(model, C*x + d .<= 0.0)

# RSOC
@constraint(model, [ii in 1:nrsoc], [dot(m1[ii],x) + b1[ii]; dot(m2[ii],x) + b2[ii]; m3[ii]*x + b3[ii]] in RotatedSecondOrderCone())
@objective(model, Min, h'*x)
optimize!(model)
println(objective_value(model))

# %% Now, solve the dual
model = Model(Gurobi.Optimizer)
@variable(model, lambda[1:neq])
@variable(model, mu[1:nineq],      lower_bound = 0.0)
@variable(model, s1[1:length(m1)], lower_bound = 0.0)
@variable(model, s2[1:length(m2)], lower_bound = 0.0)
s = Dict(ii => @variable(model, [1:size(m3[ii],1)]) for ii in 1:length(m3))

@constraint(model, [ii in 1:length(m1)], [s1[ii]; s2[ii]; s[ii]] in RotatedSecondOrderCone())
@constraint(model, h + A'*lambda + C'*mu - sum(s1[ii]*m1[ii]' + s2[ii]*m2[ii]' + m3[ii]'*s[ii] for ii in 1:length(m1))  .== 0.0)
@objective(model, Max, lambda'*b + mu'*d - sum(s1[ii]*b1[ii]  + s2[ii]*b2[ii]  + s[ii]'*b3[ii] for ii in 1:length(m1)))

optimize!(model)







# %% Now, solve the dual when the line status is parameterized by load
include("./functions.jl")

model = Model(Ipopt.Optimizer)
@variable(model, pd0_var[1:nb])
@variable(model, qd0_var[1:nb])
@variable(model, risk[1:nl])
@variable(model, alpha[1])

# this only works if pd0 and qd0 are positive
@constraint(model, 0.9*pd0 .<= pd0_var  .<= 1.1*pd0)
@constraint(model, 0.9*qd0 .<= qd0_var  .<= 1.1*qd0)
@constraint(model, 0.0      <= risk      <= 1.0)
@constraint(model, 0.25    .<= alpha    .<= 0.85)

# => sig(x) = @. 1/(1+exp(-x))
# => Ann = randn(nl, 2*nb)
# => bnn = randn(nl)
# => zl = sig(Ann*[pd0_var; qd0_var] + bnn)

x = [pd0_var; qd0_var; risk; alpha]
predictor = MathOptAI.PytorchModel(joinpath(@__DIR__, "trained_model.pt"))
zl, _ = MathOptAI.add_predictor(model, predictor, x)

#σ(x) = 1 / (1 + exp(-x))



A, b, C, d, h, m1, b1, m2, b2, m3, b3 = canonicalize(zl, pd0_var, qd0_var, nb, nl, ng, Tpfr, TpRfr, TpIfr, Tpto, TpRto, TpIto, Tqfr, TqRfr, TqIfr, Tqto, TqRto, TqIto, Eg, Efr, Eto, vmin, vmax, wr_min, wr_max, wi_min, wi_max, pg_max, qg_min, qg_max, Gs, Bs_pos, Bs_neg, smax, clin) 

@variable(model, lambda[1:neq])
@variable(model, mu[1:nineq],      lower_bound = 0.0)
@variable(model, s1[1:length(m1)], lower_bound = 0.0)
@variable(model, s2[1:length(m2)], lower_bound = 0.0)
s = Dict(ii => @variable(model, [1:size(m3[ii],1)]) for ii in 1:length(m3))
@constraint(model, [ii in 1:length(m1)],  dot(s[ii],s[ii]) <= 2*s1[ii]*s2[ii])
@constraint(model, h + A'*lambda + C'*mu - sum(s1[ii]*m1[ii]' + s2[ii]*m2[ii]' + m3[ii]'*s[ii] for ii in 1:length(m1))  .== 0.0)
@objective(model, Max, lambda'*b + mu'*d - sum(s1[ii]*b1[ii]  + s2[ii]*b2[ii]  + s[ii]'*b3[ii] for ii in 1:length(m1)))

optimize!(model)
objective_value(model)