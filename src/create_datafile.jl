#----------------------------------------------------------------------
using LinearSOC
using PGLib, Random, HDF5
using Gurobi
using JuMP

# Optimizer
gurobi_optimizer = Gurobi.Optimizer

# PGLib model
model_name = "case14_ieee" 

# Filename (WILL OVERWRITE)
h5write_filename = "data_file_14bus.h5"

# Number of datasets
n_test = 10000
n_train = 80000
n_val = 10000

# Hyperparams
alpha_min = 0.25
alpha_max = 0.85
perturb_percent = 0.50

# Debug?
debug = true

#----------------------------------------------------------------------
data = pglib(model_name)
data_copy = pglib(model_name)

total_samples = n_test + n_train + n_val    

Random.seed!(1234)

function generate_pd_qd!(data, data_copy)
    for (id, comp) in data["load"]
        comp["qd"] = data_copy["load"][id]["qd"]*(1-perturb_percent) + (2 * perturb_percent)*data_copy["load"][id]["qd"] * rand()
        comp["pd"] = data_copy["load"][id]["pd"]*(1-perturb_percent) + (2 * perturb_percent)*data_copy["load"][id]["pd"] * rand()
    end
    return data
end

file = h5open(h5write_filename, "w")
write_dataset(file, "alpha_max", alpha_max)
write_dataset(file, "alpha_min", alpha_min)
write_dataset(file, "total_samples", total_samples)
write_dataset(file, "perturb_percent", perturb_percent)

g_test = create_group(file, "test_data")
write_dataset(g_test, "index", [1])
write_dataset(g_test, "num_samples", [n_test])
for i in 1:n_test
    alpha = alpha_min + (alpha_max - alpha_min) * rand()
    generate_risk!(data, alpha)
    generate_pd_qd!(data, data_copy)

    group = create_group(g_test, string(i))

    # Load data
    load = create_group(group, "load")
    load_size = length(keys(data["bus"]))
    (qd_vals, pd_vals) = (zeros(Float32, load_size), zeros(Float32, load_size))
    for (key, value) in data["load"]
        qd_vals[value["load_bus"]] = value["qd"]
        pd_vals[value["load_bus"]] = value["pd"]
    end
    write_dataset(load, "qd", qd_vals)
    write_dataset(load, "pd", pd_vals)


    # Branch data
    branch = create_group(group, "branch")
    branch_size = length(keys(data["branch"]))
    prisk = Array{Float32}(undef, branch_size)
    for (key, value) in data["branch"]
        prisk[parse(Int, key)] = data["branch"][key]["power_risk"]
    end
    write_dataset(branch, "power_risk", prisk)
        
    # Alpha
    write_dataset(group, "alpha", [data["risk_weight"]])
end

g_train = create_group(file, "train_data")
write_dataset(g_train, "index", [1])
write_dataset(g_train, "num_samples", [n_train])
for i in 1:n_train
    alpha = alpha_min + (alpha_max - alpha_min) * rand()
    generate_risk!(data, alpha)
    generate_pd_qd!(data, data_copy)

    group = create_group(g_train, string(i))

    # Load data
    load = create_group(group, "load")
    load_size = length(keys(data["bus"]))
    (qd_vals, pd_vals) = (zeros(Float32, load_size), zeros(Float32, load_size))
    for (key, value) in data["load"]
        qd_vals[value["load_bus"]] = value["qd"]
        pd_vals[value["load_bus"]] = value["pd"]
    end
    write_dataset(load, "qd", qd_vals)
    write_dataset(load, "pd", pd_vals)

    # Branch data
    branch = create_group(group, "branch")
    branch_size = length(keys(data["branch"]))
    prisk = Array{Float32}(undef, branch_size)
    for (key, value) in data["branch"]
        prisk[parse(Int, key)] = data["branch"][key]["power_risk"]
    end
    write_dataset(branch, "power_risk", prisk)
    
    # Alpha
    write_dataset(group, "alpha", [data["risk_weight"]])
end

g_val = create_group(file, "val_data")
write_dataset(g_val, "index", [1])
write_dataset(g_val, "num_samples", [n_val])
for i in 1:n_val
    alpha = alpha_min + (alpha_max - alpha_min) * rand()
    generate_risk!(data, alpha)
    generate_pd_qd!(data, data_copy)

    group = create_group(g_val, string(i))

    # Load data
    load = create_group(group, "load")
    load_size = length(keys(data["bus"]))
    (qd_vals, pd_vals) = (zeros(Float32, load_size), zeros(Float32, load_size))
    for (key, value) in data["load"]
        qd_vals[value["load_bus"]] = value["qd"]
        pd_vals[value["load_bus"]] = value["pd"]
    end
    write_dataset(load, "qd", qd_vals)
    write_dataset(load, "pd", pd_vals)

    # Branch data
    branch = create_group(group, "branch")
    branch_size = length(keys(data["branch"]))
    prisk = Array{Float32}(undef, branch_size)
    for (key, value) in data["branch"]
        prisk[parse(Int, key)] = data["branch"][key]["power_risk"]
    end
    write_dataset(branch, "power_risk", prisk)
    
    # Alpha
    write_dataset(group, "alpha", [data["risk_weight"]])
end
close(file)