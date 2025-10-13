using PGLib
using PowerModels
using JuMP, Ipopt
using Plots
using Gurobi
using Zygote, LinearAlgebra, SparseArrays

# %% ================
include("./functions.jl")
include("./structs.jl")

# call the test case
data = pglib("pglib_opf_case118_ieee.m")
data = pglib("case24_ieee")

# test ===
#vm, va, pg, qg, Eg, Efr, Eto, g, g_fr, g_to, b, b_fr, b_to, tm, ta, pd, qd, gs, bs, nb, nl, ng, 
#vmax, vmin, pg_max, pg_min, qg_max, qg_min, smax, clin, fr_buses, to_buses = sovle_acopf_directly(data, false)

gm = parse_PM_to_SOCGridModel(data; perturb=false)

# loads
pd0 = copy(gm.pd)
qd0 = copy(gm.qd)

# %% build soc_relaxation
nl = gm.nl
nb = gm.nb
ng = gm.ng
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
@constraint(model, p_fr .- zl.*(gm.Tpfr*w + gm.TpRfr*wr + gm.TpIfr*wi) .== 0.0)
@constraint(model, p_to .- zl.*(gm.Tpto*w + gm.TpRto*wr + gm.TpIto*wi) .== 0.0)
@constraint(model, q_fr .- zl.*(gm.Tqfr*w + gm.TqRfr*wr + gm.TqIfr*wi) .== 0.0)
@constraint(model, q_to .- zl.*(gm.Tqto*w + gm.TqRto*wr + gm.TqIto*wi) .== 0.0)

@constraint(model,  -gm.Eg*pg + diagm(pd0)*zd + gm.Efr'*p_fr + gm.Eto'*p_to + pgs               .== 0.0)
@constraint(model,  -gm.Eg*qg + diagm(qd0)*zd + gm.Efr'*q_fr + gm.Eto'*q_to - qbs_pos - qbs_neg .== 0.0)

# ================================================ #
# Define the inequality constraints
@constraint(model, 0.0 .- zd .<= 0.0)
@constraint(model, zd .- 1.0 .<= 0.0)

@constraint(model, gm.vmin.^2 .- w .<= 0.0)
@constraint(model, w .- gm.vmax.^2 .<= 0.0)

@constraint(model, gm.wr_min .- wr .<= 0.0)
@constraint(model, wr .- gm.wr_max .<= 0.0)

@constraint(model, gm.wi_min .- wi .<= 0.0)
@constraint(model, wi .- gm.wi_max .<= 0.0)

@constraint(model, 0.0 .- pg    .<= 0.0)
@constraint(model, pg .- gm.pg_max .<= 0.0)

@constraint(model, gm.qg_min .- qg .<= 0.0)
@constraint(model, qg .- gm.qg_max .<= 0.0)

@constraint(model, -pgs               .<= 0.0)
@constraint(model, pgs - gm.Gs*w         .<= 0.0)
@constraint(model, -qbs_pos           .<= 0.0)
@constraint(model, qbs_pos - gm.Bs_pos*w .<= 0.0)
@constraint(model, gm.Bs_neg*w - qbs_neg .<= 0.0)
@constraint(model, qbs_neg            .<= 0.0)

# ================================================ #
# Define the RSOC constraints
@constraint(model, p_fr.^2 .+ q_fr.^2 .<= gm.smax.^2)
@constraint(model, p_to.^2 .+ q_to.^2 .<= gm.smax.^2)
@constraint(model, (wr).^2 .+ (wi).^2 .<= (gm.Efr*w).*(gm.Eto*w))

# ================================================ #
# objective!
d1 = diagm(pd0)*zd - pd0
d2 = pgs -  gm.Gs*w
d3 = qbs_pos - gm.Bs_pos*w
d4 = gm.Bs_neg*w - qbs_neg
@objective(model, Min, d1'*d1 + d2'*d2 + d3'*d3 + d4'*d4 + gm.clin'*pg)
optimize!(model)
println(objective_value(model))

# %% canonicalize
# Ax + b  = 0
# Cx + d <= 0
# s in K
# => x = [w; wr; wi; zd; pg; qg; pgs; qbs_pos; qbs_neg; p_fr; p_to; p_fr; q_to]
include("./functions.jl")

zl = 1.0
lp, soc = canonicalize(zl, pd0, qd0, gm; include_cost=true) 

# %% =================
neq   = size(lp[:A],1)
nineq = size(lp[:C],1)
nvar  = size(lp[:A],2)
nrsoc = length(soc[:m1])

# solve the model
model = Model(Gurobi.Optimizer)
@variable(model, x[1:nvar])
@constraint(model, lp[:A]*x + lp[:b] .== 0.0)
@constraint(model, lp[:C]*x + lp[:d] .<= 0.0)

# RSOC
@constraint(model, [ii in 1:nrsoc], [dot(soc[:m1][ii],x) + soc[:b1][ii]; dot(soc[:m2][ii],x) + soc[:b2][ii]; soc[:m3][ii]*x + soc[:b3][ii]] in RotatedSecondOrderCone())
@objective(model, Min, lp[:h]'*x)
optimize!(model)
println(objective_value(model))

# %% Now, solve the dual
model = Model(Gurobi.Optimizer)
@variable(model, lambda[1:neq])
@variable(model, mu[1:nineq], lower_bound = 0.0)
@variable(model, s1[1:nrsoc], lower_bound = 0.0)
@variable(model, s2[1:nrsoc], lower_bound = 0.0)
s = Dict(ii => @variable(model, [1:size(soc[:m3][ii],1)]) for ii in 1:nrsoc)

@constraint(model, [ii in 1:nrsoc], [s1[ii]; s2[ii]; s[ii]] in RotatedSecondOrderCone())
@constraint(model, lp[:h] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc)  .== 0.0)
@objective(model, Max, lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc))
optimize!(model)

println()
println(objective_value(model))

# %% now, try with pdhg
using Zygote

lambda_sol = value.(lambda)
mu_sol     = value.(mu)
s1_sol     = value.(s1)
s2_sol     = value.(s2)
s_sol      = Dict(ii => value.(s[ii]) for ii in 1:nrsoc)
x_sol      = value.(x)
nlam       = length(lambda_sol)
nmu        = length(mu_sol)
ns1        = length(s1_sol)
ns2        = length(s2_sol)
ns         = length(s_sol)
nx         = length(x_sol)

idx = Dict(:lambda => 0                .+ (1:nlam),
           :mu     => nlam             .+ (1:nmu),
           :s1     => nlam+nmu         .+ (1:ns1),
           :s2     => nlam+nmu+ns1     .+ (1:ns2))

idx_no_lam = Dict(:mu     => 0           .+ (1:nmu),
                  :s1     => nmu         .+ (1:ns1),
                  :s2     => nmu+ns1     .+ (1:ns2))

function Lagrangian(x, duals, s, idx, lp, soc)
    # compute the Lagrangian
    lambda = @view duals[idx[:lambda]]
    mu     = @view duals[idx[:mu]]
    s1     = @view duals[idx[:s1]]
    s2     = @view duals[idx[:s2]]

    c = lp[:h] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:length(s))
    L = dot(c,x) + dot(lambda,lp[:b]) + dot(mu,lp[:d]) - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + dot(s[ii],soc[:b3][ii]) for ii in 1:length(s))
    
    return L
end

function Lagrangian_implicit(x, duals_no_lam, s, idx_no_lam, lp, soc, M)
    # compute the Lagrangian
    mu     = @view duals_no_lam[idx_no_lam[:mu]]
    s1     = @view duals_no_lam[idx_no_lam[:s1]]
    s2     = @view duals_no_lam[idx_no_lam[:s2]]

    c = lp[:h] + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:length(s))
    lambda = M*(-c)

    L = dot(c,x) + dot(lambda,lp[:b]) + dot(mu,lp[:d]) - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + dot(s[ii],soc[:b3][ii]) for ii in 1:length(s))
    
    return L
end
# %%

function project_rsoc(s1,s2,s)
    if (s1 < 0.0) || (s2 < 0.0)
        s1 = 0
        s2 = 0
        s .= 0
    elseif 2*s1*s2 >= dot(s,s)
        # all good
    else # conic violation!
        d = dot(s,s) - 2*s1*s2
        alpha1 = sqrt((2*s1*s2 + d/2)/(2*s1*s2))
        s1 = s1*alpha1
        s2 = s2*alpha1

        alpha2 = (dot(s,s) - d/2)/dot(s,s)
        s = s.*sqrt(alpha2)
    end

    return s1, s2, s
end

# %% ========
xx    = zeros(nx)
xx[1:(gm.nb+gm.nl)] .= 1.0
duals = 0.1*ones(nlam+nmu+ns1+ns2)
sd    = copy(s_sol)

# %%
include("./functions.jl")

lp, soc = canonicalize(zl, pd0, qd0, gm; include_cost=true, use_float64=true) 

# %%
nits = 1000
xx    = zeros(nx)
xx[1:(gm.nb+gm.nl)] .= 1.0
duals = 0.1*ones(nlam+nmu+ns1+ns2)
sd    = copy(s_sol)
obj = zeros(nits)

eta1 = 0.1
eta2 = 0.0000000001

for ii in 1:nits

    grad_d = Zygote.gradient(DUALS -> Lagrangian(xx, DUALS, sd, idx, lp, soc), duals)[1]
    grad_s = Zygote.gradient(SD -> Lagrangian(xx, duals, SD, idx, lp, soc), sd)[1]

    duals = duals + eta1*grad_d
    for ii in 1:nrsoc
        sd[ii] = sd[ii] + eta1*grad_s[ii]
    end

    # loop and project
    for ii in 1:nrsoc
        duals[idx[:s1]][ii],duals[idx[:s2]][ii],sd[ii] = project_rsoc(duals[idx[:s1]][ii],duals[idx[:s2]][ii],sd[ii])
    end

    duals[idx[:mu]] = max.(duals[idx[:mu]], 0.0)
    duals[idx[:s1]] = max.(duals[idx[:s1]], 0.0)
    duals[idx[:s2]] = max.(duals[idx[:s2]], 0.0)

    grad_x = Zygote.gradient(X -> Lagrangian(X, duals, sd, idx, lp, soc), xx)[1]
    xx = xx - eta2*grad_x

    obj[ii] = Lagrangian(xx, duals, sd, idx, lp, soc)
    println(obj[ii])
end

# %% test with implicit lambda
M = inv(At'*At)*(At')
nits = 1000
xx   = zeros(nx)
xx[1:(gm.nb+gm.nl)] .= 1.0
duals_no_lam = 0.1*ones(nmu+ns1+ns2)
sd    = copy(s_sol)
obj = zeros(nits)

eta1 = 0.0001
eta2 = 0.0000000001


for ii in 1:nits

    grad_d = Zygote.gradient(DUALS -> Lagrangian_implicit(xx, DUALS, sd, idx_no_lam, lp, soc, M), duals_no_lam)[1]
    grad_s = Zygote.gradient(SD -> Lagrangian_implicit(xx, duals, SD, idx_no_lam, lp, soc, M), sd)[1]

    duals_no_lam = duals_no_lam + eta1*grad_d
    for ii in 1:nrsoc
        sd[ii] = sd[ii] + eta1*grad_s[ii]
    end

    # loop and project
    for ii in 1:nrsoc
        duals_no_lam[idx_no_lam[:s1]][ii],duals_no_lam[idx_no_lam[:s2]][ii],sd[ii] = project_rsoc(duals_no_lam[idx_no_lam[:s1]][ii],duals[idx_no_lam[:s2]][ii],sd[ii])
    end

    duals_no_lam[idx_no_lam[:mu]] = max.(duals_no_lam[idx_no_lam[:mu]], 0.0)
    duals_no_lam[idx_no_lam[:s1]] = max.(duals_no_lam[idx_no_lam[:s1]], 0.0)
    duals_no_lam[idx_no_lam[:s2]] = max.(duals_no_lam[idx_no_lam[:s2]], 0.0)

    grad_x = Zygote.gradient(X -> Lagrangian_implicit(X, duals_no_lam, sd, idx_no_lam, lp, soc, M), xx)[1]
    xx = xx - eta2*grad_x

    obj[ii] = Lagrangian_implicit(xx, duals_no_lam, sd, idx_no_lam, lp, soc, M)
    println(obj[ii])
end

duals_no_lam

duals = 0.1*ones(nmu+ns1+ns2)

# %% test implict lambda via ipopt
model = Model(Gurobi.Optimizer)
@variable(model, lambda[1:neq])
@variable(model, mu[1:nineq], lower_bound = 0.0)
@variable(model, s1[1:nrsoc], lower_bound = 0.0)
@variable(model, s2[1:nrsoc], lower_bound = 0.0)
s = Dict(ii => @variable(model, [1:size(soc[:m3][ii],1)]) for ii in 1:nrsoc)

@constraint(model, [ii in 1:nrsoc], [s1[ii]; s2[ii]; s[ii]] in RotatedSecondOrderCone())
#c = lp[:h] + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc)
#lambda = M*(-c)



@constraint(model, lp[:h] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc)  .== 0.0)
@objective(model, Max, lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc))
optimize!(model)

println()
println(objective_value(model))

# %% ===
m3 = min_loadshed_soc_dual(gm_test, zl0; solver=:Gurobi);

# %% ===============
include("./functions.jl")

m_soc_zrelax, m_acopf_zrelax, m_soc_zsnap, m_acopf_zsnap = compute_loadsheds(gm, m4);

# %% =======INIT==========
zl0, logit_zl0 = line_status(gm, bounds, nn_model)
dual_soln      = min_loadshed_soc_dual(gm, zl0)
pd0            = copy(gm.pd)
qd0            = copy(gm.qd)
lp, soc        = canonicalize(zl, pd0, qd0, gm)
neq            = size(lp[:A],1)
nineq          = size(lp[:C],1)
nvar           = size(lp[:A],2)
nrsoc          = length(soc[:m1])
init           = true

# %% ======PROBELM 1 (GUROBI)==========
if init == true
    zl  = ones(gm.nl)
    pd0 = copy.(gm.pd)
    qd0 = copy.(gm.qd)
    init = false
else
    zl  = value.(model_I[:zl])
    pd0 = value.(model_I[:zl])
    qd0 = value.(model_I[:zl])
end
lp, soc = canonicalize(zl, pd0, qd0, gm)
model_G = Model(Gurobi.Optimizer)
@variable(model_G, lambda[1:neq])
@variable(model_G, mu[1:nineq], lower_bound = 0.0)
@variable(model_G, s1[1:nrsoc], lower_bound = 0.0)
s = Dict(ii => @variable(model_G, [1:size(soc[:m3][ii],1)]) for ii in 1:nrsoc)
model_G[:s] = s
@variable(model_G, s2[1:nrsoc], lower_bound = 0.0)
@constraint(model_G, [ii in 1:nrsoc], [s1[ii]; s2[ii]; s[ii]] in RotatedSecondOrderCone())
@constraint(model_G, lp[:H] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc) .== 0.0)
@objective(model_G, Max, lp[:h] + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc))
optimize!(model_G)
println(objective_value(model_G))

# %% ======PROBELM 2 (IPOPT)==========
model_I = Model(Ipopt.Optimizer)
set_optimizer_attribute(model_I, "max_wall_time",  30.0)
set_attribute(model_I, "hsllib", HSL_jll.libhsl_path)
set_attribute(model_I, "linear_solver", "ma97")
@variable(model_I, pd0_var[1:gm.nb])
@variable(model_I, qd0_var[1:gm.nb])
@variable(model_I, risk[1:gm.nl])
@variable(model_I, alpha)
p_lb, p_ub = scale_load(gm.pd, bounds[:load_scale_lb], bounds[:load_scale_ub])
q_lb, q_ub = scale_load(gm.qd, bounds[:load_scale_lb], bounds[:load_scale_ub])
@constraint(model_I, p_lb              .<= pd0_var .<= p_ub)
@constraint(model_I, q_lb              .<= qd0_var .<= q_ub)
@constraint(model_I, bounds[:risk_lb]  .<=  risk   .<= bounds[:risk_ub])
@constraint(model_I, bounds[:alpha_lb]  <=  alpha   <= bounds[:alpha_ub])
x = [pd0_var; qd0_var; risk; alpha]
predictor = MathOptAI.PytorchModel(joinpath(pwd(), "models/"*nn_model))
logit_zl, _ = MathOptAI.add_predictor(model_I, predictor, x; gray_box=true) #; gray_box = true)#; reduced_space = true)#; gray_box = true)
sig(x) = 1.0 / (1.0 + exp(-x))
@variable(model_I, zl[1:gm.nl])
@constraint(model_I, zl .== sig.(logit_zl) )

# now, canonicalize
lp, soc = canonicalize(zl, pd0_var, qd0_var, gm) 
lambda = value.(model_G[:lambda])
mu     = value.(model_G[:mu])
s1     = value.(model_G[:s1])
s2     = value.(model_G[:s2])
s      = Dict(ii => value.(model_G[:s][ii]) for ii in 1:nrsoc)
@variable(model_I, G[1:length(lp[:H])])
@constraint(model_I, G*sum(pd0_var) .== lp[:H])
g = 1.0

EE = G + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc)
#@constraint(model_I, EE[1:(4*gm.nl+2*gm.nb)] .== 0.0)
@constraint(model_I, EE[[1:24; 101:124]] .== 0.0)

@objective(model_I, Max, g + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc))

set_start_value.(pd0_var, gm.pd)
set_start_value.(qd0_var, gm.qd)
set_start_value.(risk, 0.5*(bounds[:risk_lb] + bounds[:risk_ub])*ones(gm.nl))
set_start_value(alpha, 0.5*(bounds[:alpha_lb] + bounds[:alpha_ub]))
set_start_value.(zl, zl0)
set_start_value.(logit_zl, logit_zl0)
lp0, _ = canonicalize(zl0, gm.pd, gm.qd, gm) 
set_start_value.(G, value.(lp0[:H])) # no need to normalize!
optimize!(model_I)
println(objective_value(model_I))


# %%

for ii in 150:415
    println(EE[ii])
    println()
    sleep(0.01)
end


# %% 
using JuMP
using Ipopt
using LinearAlgebra

c = randn(10)
A = randn(5,10)
b = randn(5)
C = [Matrix(I,10,10);
    -Matrix(I,10,10)]
ub = ones(10)
lb = -ub
d = [-ub; lb]

# %% solve the primal!!
model = Model(Ipopt.Optimizer)
@variable(model, x[1:10])

@constraint(model, A*x + b .== 0)
@constraint(model, C*x + d .<= 0)
@objective(model, Min, dot(c,x))

optimize!(model)
println(objective_value(model))

# %% solve the dual!!
model = Model(Ipopt.Optimizer)

@variable(model, mu[1:20], lower_bound = 0.0)
@variable(model, lambda[1:5])

@constraint(model, c + A'*lambda + C'*mu .== 0)

@objective(model, Max, dot(lambda,b) + dot(mu,d))
optimize!(model)
println(objective_value(model))




