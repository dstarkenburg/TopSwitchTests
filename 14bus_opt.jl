using LinearSOC
using PGLib, Test, Random
using Gurobi
using JuMP

gurobi_optimizer = Gurobi.Optimizer

data = pglib("case14_")
data["power_flow_linearization_points"]=7
data["voltage_linearization_points"]=7
Random.seed!(1234)

generate_risk!(data, 0.2)

solution = solve_ops(data, gurobi_optimizer)
solution["termination_status"] == JuMP.LOCALLY_SOLVED

#num_branches = length(solution["solution"]["branch"])

for (key, value) in solution["solution"]["branch"]
    if solution["solution"]["branch"][key]["br_status"] == 1.0
        println("Branch: " * key *
                ", Status: " * string(solution["solution"]["branch"][key]["br_status"]) *
                ", Risk weight: " * string(data["risk_weight"]) *
                ", Power risk: " * string(data["branch"][key]["power_risk"]))
    else
end