using LinearSOC
using PGLib, Test, Random
using Gurobi
using JuMP
using Printf

gurobi_optimizer = Gurobi.Optimizer

data = pglib("case14_")
data["power_flow_linearization_points"]=7
data["voltage_linearization_points"]=7
Random.seed!(1234)

generate_risk!(data, 0.2)

solution = solve_ops(data, gurobi_optimizer)
solution["termination_status"] == JuMP.LOCALLY_SOLVED

#num_branches = length(solution["solution"]["branch"])
csvfile = open("branches.csv","w")
println("Risk weight: " * string(data["risk_weight"]))
write(csvfile, "alpha, branch, status, prisk\n")
write(csvfile, string(data["risk_weight"]), "", "", "", "", "")
for (key, value) in solution["solution"]["branch"]
    @printf("Branch: %2i, Status: %2i, Power Risk: %05.2f\n", parse(Int8, key),
             solution["solution"]["branch"][key]["br_status"],
             data["branch"][key]["power_risk"])
    temp = tuple("", parse(Int8, key),
                    solution["solution"]["branch"][key]["br_status"],
                     data["branch"][key]["power_risk"])
    write(csvfile, join(temp, ","), "\n")
end
close(csvfile)
csvfile = open("loads.csv","w")
write(csvfile, "load, status, qd, pd\n")
for (key, value) in solution["solution"]["load"]
    @printf("Load: %2i, Status: %2i, Qd: %05.2f, Pd: %05.2f\n", parse(Int8, key),
                    solution["solution"]["load"][key]["status"],
                    solution["solution"]["load"][key]["qd"],
                    solution["solution"]["load"][key]["pd"])
    temp = tuple(parse(Int8, key),
                    solution["solution"]["load"][key]["status"],
                    solution["solution"]["load"][key]["qd"],
                    solution["solution"]["load"][key]["pd"])
    write(csvfile, join(temp, ","), "\n")
end
close(csvfile)