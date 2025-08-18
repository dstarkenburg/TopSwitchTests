using PGLib, Random
using Gurobi
using JuMP
using Printf
using HDF5

##################################################################
# STOP      STOP       STOP        STOP        STOP        STOP
# This file can take HOURS to run, it is built to hopefully
# save between sample solves but if it is force stopped at the right moment
# you will corrupt your HDF5 file. Run with caution and BACKUP YOUR FILE!
##################################################################

# PGLib model
model_name = "case14_ieee"
# Name of the file generated using create_datafile.jl
output_file = "data_file_14bus.h5"

# File will keep track of how many of the total samples have been solved
# automatically by default (-1).
# Change this below to essentially solve a certain amount and stop 
# (useful if you can only run the program for a few hours and are 
# afraid of abrupt program exit corrupting your file).
# i.e. number_to_solve = 1000 will solve 1000 unsolved samples using gurobi_optimizer
# in the file, and save. It will pickup at the next unsolved sample when you rerun this file.
global number_to_solve = -1

##################################################################
h5open(output_file, "r+") do file
    global test_index = file["test_data"]["index"][1]
    global test_sample_total = file["test_data"]["num_samples"][1]
    global train_index = file["train_data"]["index"][1]
    global train_sample_total = file["train_data"]["num_samples"][1]
    global val_index = file["val_data"]["index"][1]
    global val_sample_total = file["val_data"]["num_samples"][1]
end

data = pglib(model_name)
generate_risk!(data, 1)
gurobi_optimizer = Gurobi.Optimizer


while (test_index != test_sample_total + 1 && number_to_solve != 0)
    h5open(output_file, "r+") do file
        for (id, comp) in data["load"]
            comp["qd"] = file["test_data"][string(test_index)]["load"]["qd"][parse(Int, id)]
            comp["pd"] = file["test_data"][string(test_index)]["load"]["pd"][parse(Int, id)]
        end
        data["risk_weight"] = file["test_data"][string(test_index)]["alpha"][1]
        for (id, comp) in data["branch"]
            comp["power_risk"] = file["test_data"][string(test_index)]["branch"]["power_risk"][parse(Int, id)]
        end
    end
    
    solution = solve_ops(data, gurobi_optimizer)

    size = length(solution["solution"]["branch"])
    b_status = Array{Float32}(undef, size)
    for (key, value) in solution["solution"]["branch"]
        b_status[parse(Int, key)] = value["br_status"]
    end
    h5open(output_file, "r+") do file
        write_dataset(file["test_data"][string(test_index)]["branch"], "status", b_status)
        global test_index += 1
        file["test_data"]["index"][1] = test_index
    end

    if (number_to_solve != -1 && number_to_solve != 0)
        global number_to_solve -= 1
    end
end

while (train_index != train_sample_total + 1 && number_to_solve != 0)
    h5open(output_file, "r+") do file
        for (id, comp) in data["load"]
            comp["qd"] = file["train_data"][string(train_index)]["load"]["qd"][parse(Int, id)]
            comp["pd"] = file["train_data"][string(train_index)]["load"]["pd"][parse(Int, id)]
        end
        data["risk_weight"] = file["train_data"][string(train_index)]["alpha"][1]
        for (id, comp) in data["branch"]
            comp["power_risk"] = file["train_data"][string(train_index)]["branch"]["power_risk"][parse(Int, id)]
        end
    end
        
    solution = solve_ops(data, gurobi_optimizer)

    size = length(solution["solution"]["branch"])
    b_status = Array{Float32}(undef, size)
    for (key, value) in solution["solution"]["branch"]
        b_status[parse(Int, key)] = value["br_status"]
    end
    h5open(output_file, "r+") do file
        write_dataset(file["train_data"][string(train_index)]["branch"], "status", b_status)
        global train_index += 1
        file["train_data"]["index"][1] = train_index
    end

    if (number_to_solve != -1 && number_to_solve != 0)
        global number_to_solve -= 1
    end
end
    
while (val_index != val_sample_total + 1 && number_to_solve != 0)
    h5open(output_file, "r+") do file
        for (id, comp) in data["load"]
            comp["qd"] = file["val_data"][string(val_index)]["load"]["qd"][parse(Int, id)]
            comp["pd"] = file["val_data"][string(val_index)]["load"]["pd"][parse(Int, id)]
        end
        data["risk_weight"] = file["val_data"][string(val_index)]["alpha"][1]
        for (id, comp) in data["branch"]
            comp["power_risk"] = file["val_data"][string(val_index)]["branch"]["power_risk"][parse(Int, id)]
        end
    end
        
    solution = solve_ops(data, gurobi_optimizer)

    size = length(solution["solution"]["branch"])
    b_status = Array{Float32}(undef, size)
    for (key, value) in solution["solution"]["branch"]
        b_status[parse(Int, key)] = value["br_status"]
    end
    h5open(output_file, "r+") do file
        write_dataset(file["val_data"][string(val_index)]["branch"], "status", b_status)
        global val_index += 1
        file["val_data"]["index"][1] = val_index
    end

    if (number_to_solve != -1 && number_to_solve != 0)
        global number_to_solve -= 1
    end
end