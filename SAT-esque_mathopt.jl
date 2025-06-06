using JuMP
import Ipopt
import Plots

model = Model(Ipopt.Optimizer)

@variable(model, x >= 0)

@variable(model, y >= 0)

@objective(model, Max, x*y)

@constraint(model, 2x + y == 100)

print(model)

optimize!(model)

#is_solved_and_feasible(model)

termination_status(model)

#primal_status(model)

#dual_status(model)

objective_value(model)

println(value(x))
println(value(y))
