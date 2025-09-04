function pfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta)
end

function pfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. 2*(g+g_fr)*(vm_fr/(tm^2)) - g/tm*vm_to*cos(va_fr-va_to-ta) + -b/tm*vm_to*sin(va_fr-va_to-ta)
end

function pfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. - g*vm_fr/tm*cos(va_fr-va_to-ta) + -b*vm_fr/tm*sin(va_fr-va_to-ta)
end

function pfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function pfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. - g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function qfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta)
end

function qfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -2*(b+b_fr)*(vm_fr/tm^2) + b/tm*vm_to*cos(va_fr-va_to-ta) + -g/tm*vm_to*sin(va_fr-va_to-ta)
end

function qfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_fr/tm*cos(va_fr-va_to-ta) + -g*vm_fr/tm*sin(va_fr-va_to-ta)
end

function qfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. - b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function qfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function pto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. (g+g_to)*vm_to^2 - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta)
end

function pto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. - g*vm_to/tm*cos(va_to-va_fr+ta) + -b*vm_to/tm*sin(va_to-va_fr+ta)
end

function pto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. 2*(g+g_to)*vm_to - g*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_fr/tm*sin(va_to-va_fr+ta)
end

function pto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. - g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

function pto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

function qto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -(b+b_to)*vm_to^2 + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta)
end

function qto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_to/tm*cos(va_to-va_fr+ta) + -g*vm_to/tm*sin(va_to-va_fr+ta)
end

function qto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -2*(b+b_to)*vm_to + b*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_fr/tm*sin(va_to-va_fr+ta)
end

function qto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

function qto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

Jsp_two(nl, nb, fr_buses, to_buses, dfr, dto) = sparse(1:nl, fr_buses, dfr, nl, nb) + sparse(1:nl, to_buses, dto, nl, nb)

# update for JuMP: summing sparse seemed to return dense!
Jsp(nl, nb, fr_buses, to_buses, dfr, dto) = sparse([1:nl; 1:nl], [fr_buses; to_buses], [dfr; dto], nl, nb)

function build_Jacobian(fnct::String, vr::String, vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    # fnct = function
    # vr   = differentiating variable
    if fnct == "pfr" && vr == "vm"
        dfr = pfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "pfr" && vr == "va"
        dfr = pfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "pto" && vr == "vm" 
        dfr = pto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "pto" && vr == "va" 
        dfr = pto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "qfr" && vr == "vm" 
        dfr = qfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "qfr" && vr == "va" 
        dfr = qfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "qto" && vr == "vm" 
        dfr = qto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    else
        if !(fnct == "qto" && vr == "va")
            @warn("combination not recognized!")
        end
        dfr = qto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    end

    M = Jsp(nl, nb, fr_buses, to_buses, dfr, dto)

    return M
end

# call the ref
function sovle_acopf_directly(network_data, perturb)
    # build the ref
    ref = PowerModels.build_ref(network_data)[:it][pm_it_sym][:nw][nw_id_default]

    #  build a custom OPF objective function -- the key here is "_build_opf_cl"
    pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)#, PowerModels._build_opf_cl)
    OPF_soln = optimize_model!(pm, optimizer=Ipopt.Optimizer)
    println(OPF_soln["objective"])

    OPF_soln["solution"]
    nb = length(OPF_soln["solution"]["bus"])
    vm_pm = zeros(nb)
    va_pm = zeros(nb)
    for ii in 1:nb
        vm_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["vm"]
        va_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["va"]
    end

    nl = length(OPF_soln["solution"]["branch"])
    pf_pm = zeros(nl)
    qf_pm = zeros(nl)
    pt_pm = zeros(nl)
    qt_pm = zeros(nl)

    for ii in 1:nl
        pf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pf"]
        qf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qf"]
        pt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pt"]
        qt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qt"]
    end

    # ============== Try to replicate
    model = Model(Ipopt.Optimizer)
    nb = length(network_data["bus"])
    ng = length(network_data["gen"])
    nl = length(network_data["branch"])

    bus_list  = collect(1:nb)
    line_list = collect(1:nl)
    gen_list  = collect(1:ng)

    # are all lines on?
    if nl != sum([network_data["branch"][string(line)]["br_status"] for line in line_list])
        @warn("some lines are off")
    end

    # are all gens on?
    if ng != sum([network_data["gen"][string(gen)]["gen_status"] for gen in gen_list])
        @warn("some gens are off")
    end

    @variable(model, vm[1:nb])
    @variable(model, va[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pto[1:nl])
    @variable(model, pfr[1:nl])
    @variable(model, qto[1:nl])
    @variable(model, qfr[1:nl])

    # set starts
    if perturb == false
        for ii in 1:nb
            set_start_value(vm[ii], vm_pm[ii])
            set_start_value(va[ii], va_pm[ii])
        end

        for ii in 1:nl
            set_start_value(pto[ii], pf_pm[ii])
            set_start_value(pfr[ii], qf_pm[ii])
            set_start_value(qto[ii], pt_pm[ii])
            set_start_value(qfr[ii], qt_pm[ii])
        end
    else
        # perturb!!
        for ii in 1:nb
            set_start_value(vm[ii], 1.0 + 0.1*randn())
            set_start_value(va[ii], 0.0 + 0.1*randn())
        end

        for ii in 1:nl
            set_start_value(pto[ii], randn())
            set_start_value(pfr[ii], randn())
            set_start_value(qto[ii], randn())
            set_start_value(qfr[ii], randn())
        end
        for ii in 1:ng
            set_start_value(pg[ii], randn())
            set_start_value(qg[ii], randn())
        end
    end

    ref_bus = 1
    for (bus,val) in network_data["bus"]
        if val["bus_type"] == 3
            ref_bus = val["bus_i"]
        end
    end

    # generator parameters
    pg_max = [network_data["gen"][string(gen)]["pmax"] for gen in gen_list]
    pg_min = [network_data["gen"][string(gen)]["pmin"] for gen in gen_list]
    qg_max = [network_data["gen"][string(gen)]["qmax"] for gen in gen_list]
    qg_min = [network_data["gen"][string(gen)]["qmin"] for gen in gen_list]

    # map gens to buses
    Eg = zeros(nb,ng)
    clin = zeros(ng)
    c0 = zeros(ng)
    ii = 1
    for (gen,val) in network_data["gen"]
        Eg[val["gen_bus"],val["index"]] = 1
        if val["cost"] == []
            clin[val["index"]] = 0.0
            c0[val["index"]] = 0.0
        else
            clin[val["index"]] = val["cost"][1]
            c0[val["index"]] = val["cost"][2]
        end
    end
    # => push!(cl, val["cost"][1])
    # => push!(c0, val["cost"][2])
    # => push!(cg_ind, val["index"])
    # => if length(val["cost"]) == 3
    # =>     @warn("this one has quadratic terms!")
    # => end

    # network parameters
    fr_buses = [network_data["branch"][string(line)]["f_bus"] for line in line_list]
    to_buses = [network_data["branch"][string(line)]["t_bus"] for line in line_list] 
    r        = [network_data["branch"][string(line)]["br_r"]  for line in line_list] 
    x        = [network_data["branch"][string(line)]["br_x"]  for line in line_list]
    g        = real(1 ./ (r+im*x))
    b        = imag(1 ./ (r+im*x))
    ta       = [network_data["branch"][string(line)]["shift"] for line in line_list] 
    tm       = [network_data["branch"][string(line)]["tap"]   for line in line_list] 
    g_to     = [network_data["branch"][string(line)]["g_to"]  for line in line_list] 
    g_fr     = [network_data["branch"][string(line)]["g_fr"]  for line in line_list] 
    b_to     = [network_data["branch"][string(line)]["b_to"]  for line in line_list] 
    b_fr     = [network_data["branch"][string(line)]["b_fr"]  for line in line_list] 
    amax     = [network_data["branch"][string(line)]["angmax"]  for line in line_list] 
    amin     = [network_data["branch"][string(line)]["angmin"]  for line in line_list] 

    # loads
    pd = zeros(nb)
    qd = zeros(nb)
    for (load,val) in network_data["load"]
        if val["status"] == 1
            bus = val["load_bus"]
            pd[bus] += val["pd"]
            qd[bus] += val["qd"]
        end
    end

    # shunts
    gs = zeros(nb)
    bs = zeros(nb)
    for (shunt,val) in network_data["shunt"]
        if val["status"] == 1
            bus = val["shunt_bus"]
            gs[bus] += val["gs"]
            bs[bus] += val["bs"]
        end
    end

    # build the incidence matrix
    E = zeros(nl,nb)
    for ii in 1:nl
        E[ii,fr_buses[ii]] = 1.0
        E[ii,to_buses[ii]] = -1.0
    end
    Efr = (E + abs.(E))/2
    Eto = (abs.(E) - E)/2

    # constraint 1: voltage magnitudes
    vmax = [network_data["bus"][string(bus)]["vmax"] for bus in bus_list]
    vmin = [network_data["bus"][string(bus)]["vmin"] for bus in bus_list]

    # constraint 2: flow limits
    smax = [minimum([network_data["branch"][string(line)]["rate_a"];
                    network_data["branch"][string(line)]["rate_b"];
                    network_data["branch"][string(line)]["rate_b"]]) for line in line_list]

    # flows
    vm_fr = Efr*vm
    vm_to = Eto*vm
    va_fr = Efr*va
    va_to = Eto*va

    @constraint(model, pfr .== @.  (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, qfr .== @. -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, pto .== @.  (g+g_to)*vm_to^2      - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )
    @constraint(model, qto .== @. -(b+b_to)*vm_to^2      + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )

    # add constraints -- ignore angle limits
    @constraint(model, va[ref_bus] == 0.0)
    @constraint(model, vmin   .<= vm .<= vmax)
    @constraint(model, pg_min .<= pg .<= pg_max)
    @constraint(model, qg_min .<= qg .<= qg_max)

    @constraint(model, pfr.^2 + qfr.^2 .<= smax.^2 )
    @constraint(model, pto.^2 + qto.^2 .<= smax.^2 )

    @constraint(model, Eg*pg .== pd - gs.*vm.^2 + Efr'*pfr + Eto'*pto)
    @constraint(model, Eg*qg .== qd - bs.*vm.^2 + Efr'*qfr + Eto'*qto)

    @objective(model, Min, clin'*pg)
    optimize!(model)

    println(objective_value(model))

    # output useful stuff
    return value.(vm), value.(va), value.(pg), value.(qg), Eg, Efr, Eto, g, g_fr, g_to, b, b_fr, b_to, tm, ta, pd, qd, gs, bs, nb, nl, ng, vmax, vmin, pg_max, pg_min, qg_max, qg_min, smax, clin, fr_buses, to_buses
end

function test_jacobians(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pfr = Zygote.jacobian(vm0 -> pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jvm_pto = Zygote.jacobian(vm0 -> pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jvm_qfr = Zygote.jacobian(vm0 -> qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jvm_qto = Zygote.jacobian(vm0 -> qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jva_pfr = Zygote.jacobian(va0 -> pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]
    Jva_pto = Zygote.jacobian(va0 -> pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]
    Jva_qfr = Zygote.jacobian(va0 -> qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]
    Jva_qto = Zygote.jacobian(va0 -> qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]

    Jvm_pfr_n = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto_n = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr_n = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto_n = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr_n = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto_n = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr_n = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto_n = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    # compare
    println(norm(Matrix(Jvm_pfr_n) - Jvm_pfr))
    println(norm(Matrix(Jvm_pto_n) - Jvm_pto))
    println(norm(Matrix(Jvm_qfr_n) - Jvm_qfr))
    println(norm(Matrix(Jvm_qto_n) - Jvm_qto))
    println(norm(Matrix(Jva_pfr_n) - Jva_pfr))
    println(norm(Matrix(Jva_pto_n) - Jva_pto))
    println(norm(Matrix(Jva_qfr_n) - Jva_qfr))
    println(norm(Matrix(Jva_qto_n) - Jva_qto))
end

""" Solve ACOPF, using IPOPT, with a linearized ACOPF model"""
function linear_min_ipopt(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    lambda_p, lambda_q, mu_sfr, mu_sto, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin)
    # vm0 = bias point
    # va0 = bias point

    model = Model(Ipopt.Optimizer)
    nb = length(vm0)
    nl = length(g)
    ng = length(clin)
    @variable(model, dvm[1:nb])
    @variable(model, dva[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])

    @constraint(model,    vmin   .<= dvm .<= vmax    )
    @constraint(model, -ones(nb) .<= dva .<= ones(nb))
    @constraint(model, pg_min    .<= pg  .<= pg_max  )
    @constraint(model, qg_min    .<= qg  .<= qg_max  )

    @constraint(model, dva[69] == 0.0)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    pfr_approx = pfr_0 + Jvm_pfr*(dvm - vm0) + Jva_pfr*(dva - va0)
    pto_approx = pto_0 + Jvm_pto*(dvm - vm0) + Jva_pto*(dva - va0)
    qfr_approx = qfr_0 + Jvm_qfr*(dvm - vm0) + Jva_qfr*(dva - va0)
    qto_approx = qto_0 + Jvm_qto*(dvm - vm0) + Jva_qto*(dva - va0)

    pfr_sq_approx = pfr_sq_0 + Jvm_pfr_sq*(dvm - vm0) + Jva_pfr_sq*(dva - va0)
    pto_sq_approx = pto_sq_0 + Jvm_pto_sq*(dvm - vm0) + Jva_pto_sq*(dva - va0)
    qfr_sq_approx = qfr_sq_0 + Jvm_qfr_sq*(dvm - vm0) + Jva_qfr_sq*(dva - va0)
    qto_sq_approx = qto_sq_0 + Jvm_qto_sq*(dvm - vm0) + Jva_qto_sq*(dva - va0)

    # vm.^2 = 2*vm0
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 

    @objective(model, Min, L)
    optimize!(model)
    println(objective_value(model))

    return value.(dvm), value.(dva)
end

""" This function does this same thing as linear_min_ipopt(), but the Lagrangian is written
so that decision are written with their linear coefficients"""
function linear_min_ipopt_decomposed(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    lambda_p, lambda_q, mu_sfr, mu_sto, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin)
    # vm0 = bias point
    # va0 = bias point

    model = Model(Ipopt.Optimizer)
    nb = length(vm0)
    nl = length(g)
    ng = length(clin)
    @variable(model, dvm[1:nb])
    @variable(model, dva[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])

    @constraint(model,    vmin   .<= dvm .<= vmax    )
    @constraint(model, -ones(nb) .<= dva .<= ones(nb))
    @constraint(model, pg_min    .<= pg  .<= pg_max  )
    @constraint(model, qg_min    .<= qg  .<= qg_max  )

    @constraint(model, dva[69] == 0.0)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    @objective(model, Min, L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0)

    optimize!(model)
    println(objective_value(model))

    return value.(dvm), value.(dva)
end

""" solve the dual variables with IPOPT directly :) """
function linear_min_dual_norm_ipopt(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
    # vm0 = bias point
    # va0 = bias point

    nb = length(vm0)
    nl = length(g)
    ng = length(clin)

    # normalization tools
    sig_pg = 0.5*(pg_max-pg_min)
    mu_pg  = 0.5*(pg_max+pg_min)
    sig_qg = 0.5*(qg_max-qg_min)
    mu_qg  = 0.5*(qg_max+qg_min)
    sig_vm = 0.5*(vmax-vmin)
    mu_vm  = 0.5*(vmax+vmin)
    amax   =  ones(nb)
    amin   = -ones(nb)
    sig_va = 0.5*(amax-amin)
    mu_va  = 0.5*(amax+amin)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 250)
    @variable(model, lambda_p[1:nb])
    @variable(model, lambda_q[1:nb])
    @variable(model, mu_sfr[1:nl])
    @variable(model, mu_sto[1:nl])

    @constraint(model, 0.0 .<= mu_sfr)
    @constraint(model, 0.0 .<= mu_sto)
    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    # add a normalization step
    """
    L  =  L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0
       =  L_pg*(sig_pg*pg_norm+mu_pg) + L_qg*(sig_qg*qg_norm+mu_qg) + L_vm*(sig_dvm*dvm_norm+mu_dvm) + L_va*(sig_dva*dva_norm+mu_dva) + L_0
       =  L_pg*sig_pg*pg_norm+L_pg*mu_pg + L_qg*sig_qg*qg_norm+L_qg*mu_qg + L_vm*sig_dvm*dvm_norm+L_vm*mu_dvm + L_va*sig_dva*dva_norm+L_va*mu_dva + L_0
       =  (L_pg*sig_pg)*pg_norm + (L_qg*sig_qg)*qg_norm + (L_vm*sig_dvm)*dvm_norm + (L_va*sig_dva)*dva_norm + L_0+L_pg*mu_pg+L_qg*mu_qg+L_vm*mu_dvm+L_va*mu_dva
    """
    L_pg_norm = L_pg.*(sig_pg')
    L_qg_norm = L_qg.*(sig_qg')
    L_vm_norm = L_vm.*(sig_vm')
    L_va_norm = L_va.*(sig_va')
    L0_norm   = L_0 + L_pg*mu_pg + L_qg*mu_qg + L_vm*mu_vm + L_vm*mu_va

    # remove the reference bus
    #L_va_norm[69] = 0.0*L_va_norm[69]
    # deleteat!(L_va_norm_v, 69)
    # L_va_norm = L_va_norm_v'

    # solve!
    # => t = norm([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm], 1)
    x = vec([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm])
    @variable(model, t)
    @constraint(model, [t; x] in MOI.NormOneCone(1 + length(x)))


    @objective(model, Max, -t + L0_norm)

    optimize!(model)
    println(objective_value(model))

    return objective_value(model), value.(lambda_p), value.(lambda_q), value.(mu_sfr), value.(mu_sto)
end


""" Thus function takes the primal, linearizes, and solves the dual norm"""
function linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,lambda_p, 
    lambda_q, mu_sfr, mu_sto, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, 
    b, b_fr, b_to, tm, ta, clin, scale_va)
    # vm0 = bias point
    # va0 = bias point

    nb = length(vm0)
    nl = length(g)
    ng = length(clin)

    # normalization tools
    sig_pg = 0.5*(pg_max-pg_min)
    mu_pg  = 0.5*(pg_max+pg_min)
    sig_qg = 0.5*(qg_max-qg_min)
    mu_qg  = 0.5*(qg_max+qg_min)
    sig_vm = 0.5*(vmax-vmin)
    mu_vm  = 0.5*(vmax+vmin)
    amax   =  ones(nb)
    amin   = -ones(nb)
    sig_va = 0.5*(amax-amin)
    mu_va  = 0.5*(amax+amin)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    # add a normalization step
    """
    L  =  L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0
       =  L_pg*(sig_pg*pg_norm+mu_pg) + L_qg*(sig_qg*qg_norm+mu_qg) + L_vm*(sig_dvm*dvm_norm+mu_dvm) + L_va*(sig_dva*dva_norm+mu_dva) + L_0
       =  L_pg*sig_pg*pg_norm+L_pg*mu_pg + L_qg*sig_qg*qg_norm+L_qg*mu_qg + L_vm*sig_dvm*dvm_norm+L_vm*mu_dvm + L_va*sig_dva*dva_norm+L_va*mu_dva + L_0
       =  (L_pg*sig_pg)*pg_norm + (L_qg*sig_qg)*qg_norm + (L_vm*sig_dvm)*dvm_norm + (L_va*sig_dva)*dva_norm + L_0+L_pg*mu_pg+L_qg*mu_qg+L_vm*mu_dvm+L_va*mu_dva
    """
    L_pg_norm = L_pg.*(sig_pg')
    L_qg_norm = L_qg.*(sig_qg')
    L_vm_norm = L_vm.*(sig_vm')
    L_va_norm = L_va.*(sig_va')
    L0_norm   = L_0 + L_pg*mu_pg + L_qg*mu_qg + L_vm*mu_vm + L_vm*mu_va

    # remove the reference bus
    #L_va_norm[69] = 0.0*L_va_norm[69]
    # deleteat!(L_va_norm_v, 69)
    # L_va_norm = L_va_norm_v'

    # solve!
    opt_val = -norm([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm], 1) + L0_norm

    return opt_val
end

""" Compute the Lagrangian"""
function Lagrangian(pg, qg, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Eg, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin)

    # compute flows
    pfr_0 = pfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    # square flows
    pfr_sq = pfr_0.^2
    pto_sq = pto_0.^2
    qfr_sq = qfr_0.^2
    qto_sq = qto_0.^2

    L = clin'*pg + 
        lambda_p'*(pd - gs.*vm + Efr'*pfr_0 + Eto'*pto_0 - (Eg*pg)) + 
        lambda_q'*(qd - bs.*vm + Efr'*qfr_0 + Eto'*qto_0 - (Eg*qg)) + 
        mu_sfr'*(pfr_sq + qfr_sq - (smax.^2)) + 
        mu_sto'*(pto_sq + qto_sq - (smax.^2)) 

    return L
end

""" incorporate constraints on vm and va (not standard)"""
function linear_min_dual_norm_constrained(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    lambda_p, lambda_q, mu_sfr, mu_sto, lam_vm, lam_va, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
    # vm0 = bias point
    # va0 = bias point

    nb = length(vm0)
    nl = length(g)
    ng = length(clin)

    # normalization tools
    sig_pg = 0.5*(pg_max-pg_min)
    mu_pg  = 0.5*(pg_max+pg_min)
    sig_qg = 0.5*(qg_max-qg_min)
    mu_qg  = 0.5*(qg_max+qg_min)
    sig_vm = 0.5*(vmax-vmin)
    mu_vm  = 0.5*(vmax+vmin)
    amax   =  ones(nb)
    amin   = -ones(nb)
    sig_va = 0.5*(amax-amin)
    mu_va  = 0.5*(amax+amin)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    # add a normalization step
    """
    L  =  L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0 + lam_vm'*(vm0-dvm) + lam_va'*(va0-dva)
       =
       =  L_pg*pg + L_qg*qg + (L_vm-lam_vm')*dvm + (L_va-lam_va')*dva + (L_0 + lam_vm'*vm0 + lam_va'*va0)
       =
       =  L_pg*(sig_pg*pg_norm+mu_pg) + L_qg*(sig_qg*qg_norm+mu_qg) + L_vm*(sig_dvm*dvm_norm+mu_dvm) + L_va*(sig_dva*dva_norm+mu_dva) + L_0
       =  L_pg*sig_pg*pg_norm+L_pg*mu_pg + L_qg*sig_qg*qg_norm+L_qg*mu_qg + L_vm*sig_dvm*dvm_norm+L_vm*mu_dvm + L_va*sig_dva*dva_norm+L_va*mu_dva + L_0
       =  (L_pg*sig_pg)*pg_norm + (L_qg*sig_qg)*qg_norm + (L_vm*sig_dvm)*dvm_norm + (L_va*sig_dva)*dva_norm + L_0+L_pg*mu_pg+L_qg*mu_qg+L_vm*mu_dvm+L_va*mu_dva
    """
    L_pg_norm = L_pg.*(sig_pg')
    L_qg_norm = L_qg.*(sig_qg')
    L_vm_norm = (L_vm-lam_vm').*(sig_vm')
    L_va_norm = (L_va-lam_va').*(sig_va')
    L0_norm   = L_0 + L_pg*mu_pg + L_qg*mu_qg + L_vm*mu_vm + L_vm*mu_va + (lam_vm'*vm0 + lam_va'*va0)

    # solve!
    opt_val = -norm([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm], 1) + L0_norm

    return opt_val
end

function finiteDiff(epsilon, vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
    # get f0
    f0 = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)

    grad_lambda_p = 0*similar(lambda_p)
    grad_lambda_q = 0*similar(lambda_q)
    grad_mu_sfr   = 0*similar(mu_sfr)
    grad_mu_sto   = 0*similar(mu_sto)
    grad_vm       = 0*similar(vm)
    grad_va       = 0*similar(va)

    # lambda_p
    for ii in 1:length(lambda_p)
        lambda_p[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_lambda_p[ii] = (fp - f0)/epsilon
        lambda_p[ii] -= epsilon
    end

    # lambda_q
    for ii in 1:length(lambda_q)
        lambda_q[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_lambda_q[ii] = (fp - f0)/epsilon
        lambda_q[ii] -= epsilon
    end

    # grad_mu_sfr
    for ii in 1:length(mu_sfr)
        mu_sfr[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_mu_sfr[ii] = (fp - f0)/epsilon
        mu_sfr[ii] -= epsilon
    end

    # grad_mu_sto
    for ii in 1:length(mu_sto)
        mu_sto[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_mu_sto[ii] = (fp - f0)/epsilon
        mu_sto[ii] -= epsilon
    end

    # vm
    for ii in 1:length(vm)
        vm[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_vm[ii] = (fp - f0)/epsilon
        vm[ii] -= epsilon
    end

    # va
    for ii in 1:length(va)
        va[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_va[ii] = (fp - f0)/epsilon
        va[ii] -= epsilon
    end

    # output
    return grad_lambda_p, grad_lambda_q, grad_mu_sfr, grad_mu_sto, grad_vm, grad_va
end

function soc_voltage_bound_vectors(data, nl, fr_buses, to_buses)
    # build the reg
    pm_ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    wr_min_pm, wr_max_pm, wi_min_pm, wi_max_pm = ref_calc_voltage_product_bounds(pm_ref[:buspairs])

    # now, loop over the line list, and populate the updated vectors
    wr_min = zeros(nl)
    wr_max = zeros(nl)
    wi_min = zeros(nl)
    wi_max = zeros(nl)
    for ii in 1:nl
        pair = (fr_buses[ii], to_buses[ii])
        wr_min[ii] = wr_min_pm[pair]
        wr_max[ii] = wr_max_pm[pair]
        wi_min[ii] = wi_min_pm[pair]
        wi_max[ii] = wi_max_pm[pair]
    end

    return wr_min, wr_max, wi_min, wi_max
end

function canonicalize(zl, pd0, qd0, nb, nl, ng, Tpfr, TpRfr, TpIfr, Tpto, TpRto, TpIto, Tqfr, TqRfr, TqIfr, Tqto, TqRto, TqIto, Eg, Efr, Eto, vmin, vmax, wr_min, wr_max, wi_min, wi_max, pg_max, qg_min, qg_max, Gs, Bs_pos, Bs_neg, smax, clin) 
    w_idx       =                     (1:nb)
    wr_idx      = w_idx[end]       .+ (1:nl)
    wi_idx      = wr_idx[end]      .+ (1:nl)
    zd_idx      = wi_idx[end]      .+ (1:nb)
    pg_idx      = zd_idx[end]      .+ (1:ng)  
    qg_idx      = pg_idx[end]      .+ (1:ng)  
    pgs_idx     = qg_idx[end]      .+ (1:nb)
    qbs_pos_idx = pgs_idx[end]     .+ (1:nb)
    qbs_neg_idx = qbs_pos_idx[end] .+ (1:nb)
    p_fr_idx    = qbs_neg_idx[end] .+ (1:nl)
    p_to_idx    = p_fr_idx[end]    .+ (1:nl)   
    q_fr_idx    = p_to_idx[end]    .+ (1:nl)   
    q_to_idx    = q_fr_idx[end]    .+ (1:nl)
    t_idx       = q_to_idx[end]     + 1

    nvar  = t_idx[end]
    neq   = 2*nb + 4*nl
    nineq = 2*nb + 2*nb + 4*nl + 4*ng + 6*nb

    # => A = zeros(neq, nvar)
    A = Matrix{NonlinearExpr}(undef, neq, nvar)
    A .= 0.0

    b = zeros(neq) # this stays 0

    # flow constraints
    A[1:nl, p_fr_idx]          = Matrix(I,nl,nl)
    A[(nl+1):2*nl, p_to_idx]   = Matrix(I,nl,nl)
    A[(2*nl+1):3*nl, q_fr_idx] = Matrix(I,nl,nl)
    A[(3*nl+1):4*nl, q_to_idx] = Matrix(I,nl,nl)
    Mflow = [-zl.*[Tpfr TpRfr TpIfr];
             -zl.*[Tpto TpRto TpIto];
             -zl.*[Tqfr TqRfr TqIfr];
             -zl.*[Tqto TqRto TqIto]]
    A[1:4*nl,w_idx[1]:wi_idx[end]] = Mflow

    # injection constraints
    pjidx = 4*nl    .+ (1:nb)
    qjidx = 4*nl+nb .+ (1:nb)

    A[pjidx,pg_idx]   = -Eg
    A[pjidx,zd_idx]   = diagm(pd0)
    A[pjidx,p_fr_idx] = Efr'
    A[pjidx,p_to_idx] = Eto'
    A[pjidx,pgs_idx]  = Matrix(I,nb,nb)

    A[qjidx,qg_idx]   = -Eg
    A[qjidx,zd_idx]   = diagm(qd0)
    A[qjidx,q_fr_idx] = Efr'
    A[qjidx,q_to_idx] = Eto'
    A[qjidx,qbs_pos_idx]  = -Matrix(I,nb,nb)
    A[qjidx,qbs_neg_idx]  = -Matrix(I,nb,nb)

    C = zeros(nineq, nvar)
    d = zeros(nineq)

    idx_nd = 0
    C[idx_nd .+ (1:nb), zd_idx] = -Matrix(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), zd_idx]= +Matrix(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= -1.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), w_idx] = -Matrix(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= vmin.^2
    idx_nd += nb

    C[idx_nd .+ (1:nb), w_idx] = +Matrix(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= -vmax.^2
    idx_nd += nb

    C[idx_nd .+ (1:nl), wr_idx] = -Matrix(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= wr_min
    idx_nd += nl

    C[idx_nd .+ (1:nl), wr_idx] = Matrix(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -wr_max
    idx_nd += nl

    C[idx_nd .+ (1:nl), wi_idx] = -Matrix(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= wi_min
    idx_nd += nl

    C[idx_nd .+ (1:nl), wi_idx] = Matrix(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -wi_max
    idx_nd += nl

    C[idx_nd .+ (1:ng), pg_idx] = -Matrix(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= 0.0
    idx_nd += ng

    C[idx_nd .+ (1:ng), pg_idx] = Matrix(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -pg_max
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = -Matrix(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= qg_min
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = Matrix(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -qg_max
    idx_nd += ng

    C[idx_nd .+ (1:nb), pgs_idx] = -Matrix(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), pgs_idx] = Matrix(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]   = -Gs
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_pos_idx] = -Matrix(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_pos_idx] = Matrix(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = -Bs_pos
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_neg_idx] = -Matrix(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = Bs_neg
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_neg_idx] = Matrix(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    nobj = nb + nb + nb + nb
    # => F = zeros(nobj,nvar)
    F = Matrix{NonlinearExpr}(undef, nobj,nvar)
    F .= 0.0
    # => f = zeros(nobj)
    f = Vector{NonlinearExpr}(undef, nobj)
    f.= 0.0

    # d1
    F[1:nb,zd_idx] = diagm(pd0)
    f[1:nb] = -pd0

    # d2
    F[nb .+ (1:nb),pgs_idx] = Matrix(I,nb,nb)
    F[nb .+ (1:nb),w_idx]   = -Gs
    f[nb .+ (1:nb)] .= 0.0

    # d3
    F[2*nb .+ (1:nb),qbs_pos_idx] = Matrix(I,nb,nb)
    F[2*nb .+ (1:nb),w_idx]       = -Bs_pos
    f[2*nb .+ (1:nb)] .= 0.0

    # d4
    F[3*nb .+ (1:nb),qbs_neg_idx] = -Matrix(I,nb,nb)
    F[3*nb .+ (1:nb),w_idx]       = Bs_neg
    f[3*nb .+ (1:nb)] .= 0.0

    # linear terms in the objective
    h = zeros(nvar)
    h[pg_idx] = clin
    h[end]    = 1

    # apply all RSOCs
    nrsoc = 3*nl + 1

    m1 = Vector{Any}(undef,nrsoc) #[Vector{Any}(undef,10)for ii in 1:nrsoc]
    m2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    m3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    b1 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    # flow limit f -> t
    for ii in 1:nl
        m1[ii] = zeros(1,nvar)
        b1[ii] = smax[ii]^2

        m2[ii] = zeros(1,nvar)
        b2[ii] = 0.5

        m3[ii] = zeros(2,nvar)
        m3[ii][1,p_fr_idx[ii]] = 1
        m3[ii][2,q_fr_idx[ii]] = 1
        b3[ii] = zeros(2)
    end

    # flow limit t -> f
    for ii in 1:nl
        m1[ii+nl] = zeros(1,nvar)
        b1[ii+nl] = smax[ii]^2

        m2[ii+nl] = zeros(1,nvar)
        b2[ii+nl] = 0.5

        m3[ii+nl] = zeros(2,nvar)
        m3[ii+nl][1,p_to_idx[ii]] = 1
        m3[ii+nl][2,q_to_idx[ii]] = 1
        b3[ii+nl] = zeros(2)
    end

    # RSOC on voltage
    for ii in 1:nl
        m1[ii+2*nl] = zeros(1,nvar)
        m1[ii+2*nl][w_idx] = Efr[ii,:]
        b1[ii+2*nl] = 0

        m2[ii+2*nl] = zeros(1,nvar)
        m2[ii+2*nl][w_idx] = 0.5*Eto[ii,:]
        b2[ii+2*nl] = 0

        m3[ii+2*nl] = zeros(2,nvar)
        m3[ii+2*nl][1,wr_idx[ii]] = 1
        m3[ii+2*nl][2,wi_idx[ii]] = 1
        b3[ii+2*nl] = zeros(2)
    end

    # cost term!
    m1[end] = zeros(1,nvar)
    m1[end][t_idx] = 1
    b1[end] = 0

    m2[end] = zeros(1,nvar)
    b2[end] = 0.5

    m3[end] = F
    b3[end] = f

    return A, b, C, d, h, m1, b1, m2, b2, m3, b3
end