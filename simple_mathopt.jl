using JuMP
import Ipopt
import Plots

model = Model(Ipopt.Optimizer)

@variable(model, x)

@variable(model, y)

@variable(model, z >= 0.25)

@objective(model, Min, x^2 + y^2 + z^2)

@constraint(model, z == sin(x) * y)

print(model)

optimize!(model)

#is_solved_and_feasible(model)

#termination_status(model)

#primal_status(model)

#dual_status(model)

#objective_value(model)

