using LinearSOC
using PGLib, Test, Random
using Gurobi
using JuMP
using Printf
using HDF5

function generate_pd_qd!(data, data_copy)
    for (id, comp) in data["load"]
        comp["qd"] = data_copy["load"][id]["qd"]*0.25 + 1.75*data_copy["load"][id]["qd"] * rand()
        comp["pd"] = data_copy["load"][id]["pd"]*0.25 + 1.75*data_copy["load"][id]["pd"] * rand()
    end
    return data
end

gurobi_optimizer = Gurobi.Optimizer

data = pglib("case14_")
data_copy = pglib("case14_")
Random.seed!(1234)

# Number of datasets
n_test = 10000
n_train = 80000
n_val = 10000

# Hyperparams
alpha = rand([(0.0 + 0.3*rand()), (0.3 + 0.3*rand()), (0.6 + 0.3*rand())])

# Write to file
h5write_filename = "data.h5"

h5open(h5write_filename, "w") do file
    # DEBUG
    val_mixed = 0
    val_on = 0
    val_off = 0

    test_mixed = 0
    test_on = 0
    test_off = 0

    train_mixed = 0
    train_on = 0
    train_off = 0

    # Create test_data group
    count_mixed = 0
    count_all_off = 0
    count_all_on = 0
    g_test = create_group(file, "test_data")
    for i in 1:n_test
        generate_risk!(data, alpha)
        generate_pd_qd!(data, data_copy)
        solution = solve_ops(data, gurobi_optimizer)

        branch_status = [value["br_status"] for (key, value) in solution["solution"]["branch"]]
        n_on = sum(branch_status)
        total = length(branch_status)

        if n_on == 0 # All branches turned off
            count_all_off += 1
        elseif n_on == total # All branches turned ON
            count_all_on += 1
        else # Mixed branch statuses
            count_mixed += 1
        end

        group = create_group(g_test, string(i))

        # Load data
        load = create_group(group, "load")
        (qd_vals, pd_vals) = (Float32[], Float32[])
        for (key, value) in data["load"]
            push!(qd_vals, value["qd"])
            push!(pd_vals, value["pd"])
        end
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
    test_mixed += count_mixed
    test_on += count_all_on
    test_off += count_all_off
    
    # Create train_data group
    count_mixed = 0
    count_all_off = 0
    count_all_on = 0
    g_train = create_group(file, "train_data")
    for i in 1:n_train
        generate_risk!(data, alpha)
        generate_pd_qd!(data, data_copy)
        solution = solve_ops(data, gurobi_optimizer)

        branch_status = [value["br_status"] for (key, value) in solution["solution"]["branch"]]
        n_on = sum(branch_status)
        total = length(branch_status)

        if n_on == 0 # All branches turned off
            count_all_off += 1
        elseif n_on == total # All branches turned ON
            count_all_on += 1
        else # Mixed branch statuses
            count_mixed += 1
        end

        group = create_group(g_train, string(i))

        # Load data
        load = create_group(group, "load")
        (qd_vals, pd_vals) = (Float32[], Float32[])
        for (key, value) in data["load"]
            push!(qd_vals, value["qd"])
            push!(pd_vals, value["pd"])
        end
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
    train_mixed += count_mixed
    train_on += count_all_on
    train_off += count_all_off

    # Create val_data group
    count_mixed = 0
    count_all_off = 0
    count_all_on = 0
    g_val = create_group(file, "val_data")
    for i in 1:n_val
        generate_risk!(data, alpha)
        generate_pd_qd!(data, data_copy)
        solution = solve_ops(data, gurobi_optimizer)

        branch_status = [value["br_status"] for (key, value) in solution["solution"]["branch"]]
        n_on = sum(branch_status)
        total = length(branch_status)

        if n_on == 0 # All branches turned off
            count_all_off += 1
        elseif n_on == total # All branches turned ON
            count_all_on += 1
        else # Mixed branch statuses
            count_mixed += 1
        end

        group = create_group(g_val, string(i))

        # Load data
        load = create_group(group, "load")
        (qd_vals, pd_vals) = (Float32[], Float32[])
        for (key, value) in data["load"]
            push!(qd_vals, value["qd"])
            push!(pd_vals, value["pd"])
        end
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
    val_mixed += count_mixed
    val_on += count_all_on
    val_off += count_all_off
    
    # DEBUG
    println(@printf("Test On: %f", test_on))
    println(@printf("Test Off: %f", test_off))
    println(@printf("Test Mixed: %f", test_mixed))

    println(@printf("Train On: %f", train_on))
    println(@printf("Train Off: %f", train_off))
    println(@printf("Train Mixed: %f", train_mixed))

    println(@printf("Val On: %f", val_on))
    println(@printf("Val Off: %f", val_off))
    println(@printf("Val Mixed: %f", val_mixed))
end