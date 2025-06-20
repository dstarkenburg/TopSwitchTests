using LinearSOC
using PGLib, Test, Random
using Gurobi
using JuMP
using Printf
using HDF5

function generate_pd_qd!(data, data_copy)
    for (id ,comp) in data["load"]
        comp["qd"] = data_copy["load"][id]["qd"]/2 + data_copy["load"][id]["qd"]*rand()
        comp["pd"] = data_copy["load"][id]["pd"]/2 + data_copy["load"][id]["pd"]*rand()
    end
    return data
end

gurobi_optimizer = Gurobi.Optimizer

data = pglib("case14_")
data_copy = copy(data)
Random.seed!(1234)

# Number of datasets
n_test = 2000
n_train = 8000

# Write to file
h5write_filename = "data.h5"

h5open(h5write_filename, "w") do file
    # Create test_data group
    g_test = create_group(file, "test_data")
    for i in 1:n_test
        generate_risk!(data, (0.2 + (0.3)*rand()))
        generate_pd_qd!(data, data_copy)
        solution = solve_ops(data, gurobi_optimizer)
        group = create_group(g_test, "$i")

        # Load data
        (qd_vals, pd_vals) = (Float32[], Float32[])
        for (key, value) in data["load"]
            push!(qd_vals, value["qd"])
            push!(pd_vals, value["pd"])
        end
        load = create_group(group, "load")
        write_dataset(load, "qd", qd_vals)
        write_dataset(load, "pd", pd_vals)

        # Branch data
        branch = create_group(group, "branch")
        (b_status, prisk) = (Float32[], Float32[])
        for (key, value) in solution["solution"]["branch"]
            push!(b_status, value["br_status"])
            push!(prisk, data["branch"][key]["power_risk"])
        end
        write_dataset(branch, "status", b_status)
        write_dataset(branch, "power_risk", prisk)

        # Alpha
        write_dataset(group, "alpha", [data["risk_weight"]])
    end

    # Create train_data group
    g_train = create_group(file, "train_data")
    for i in 1:n_train
        generate_risk!(data, (0.2 + (0.3)*rand()))
        generate_pd_qd!(data, data_copy)
        solution = solve_ops(data, gurobi_optimizer)
        group = create_group(g_train, "$i")

        # Load data
        (qd_vals, pd_vals) = (Float32[], Float32[])
        for (key, value) in data["load"]
            push!(qd_vals, value["qd"])
            push!(pd_vals, value["pd"])
        end
        load = create_group(group, "load")
        write_dataset(load, "qd", qd_vals)
        write_dataset(load, "pd", pd_vals)

        # Branch data
        branch = create_group(group, "branch")
        (b_status, prisk) = (Float32[], Float32[])
        for (key, value) in solution["solution"]["branch"]
            push!(b_status, value["br_status"])
            push!(prisk, data["branch"][key]["power_risk"])
        end
        write_dataset(branch, "status", b_status)
        write_dataset(branch, "power_risk", prisk)

        # Alpha
        write_dataset(group, "alpha", [data["risk_weight"]])
    end
end