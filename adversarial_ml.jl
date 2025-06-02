using JuMP
import Flux
import Ipopt
import MathOptAI
import MLDatasets
import Plots

## https://lanl-ansi.github.io/MathOptAI.jl/stable/tutorials/mnist/ ##

train_data = MLDatasets.MNIST(; split = :train)

test_data = MLDatasets.MNIST(; split = :test)

function plot_image(x::Matrix; kwargs...)
    return Plots.heatmap(
        x'[size(x, 1):-1:1, :];
        xlims = (1, size(x, 2)),
        ylims = (1, size(x, 1)),
        aspect_ratio = true,
        legend = false,
        xaxis = false,
        yaxis = false,
        kwargs...,
    )
end

function plot_image(predictor, x::Matrix)
    score, index = findmax(predictor(vec(x)))
    title = "Predicted: $(index - 1) ($(round(Int, 100 * score))%)"
    return plot_image(x; title)
end

function score_model(predictor, data)
    x, y = only(data_loader(data; batchsize = length(data)))
    y_hat = predictor(x)
    is_correct = Flux.onecold(y) .== Flux.onecold(y_hat)
    p = round(100 * sum(is_correct) / length(is_correct); digits = 2)
    println("Accuracy = $p %")
    return
end

function plot_image(instance::NamedTuple)
    return plot_image(instance.features; title = "Label = $(instance.targets)")
end

function data_loader(data; batchsize, shuffle = false)
    x = reshape(data.features, 28^2, :)
    y = Flux.onehotbatch(data.targets, 0:9)
    return Flux.DataLoader((x, y); batchsize, shuffle)
end

# Plots.plot([plot_image(train_data[i]) for i in 1:6]...; layout = (2, 3))

# https://stats.stackexchange.com/questions/376312/mnist-digit-recognition-what-is-the-best-we-can-get-with-a-fully-connected-nn-o
predictor = Flux.Chain(
    Flux.Dense(28^2 => 784, Flux.sigmoid),
    Flux.Dense(784 => 400),
    Flux.Dense(400 => 400),
    Flux.Dense(400 => 10),
    Flux.softmax,
)

begin
    train_loader = data_loader(train_data; batchsize = 256, shuffle = true)
    optimizer_state = Flux.setup(Flux.Adam(3e-4), predictor)
    for epoch in 1:15
        loss = 0.0
        for (x, y) in train_loader
            loss_batch, gradient = Flux.withgradient(predictor) do model
                return Flux.crossentropy(model(x), y)
            end
            Flux.update!(optimizer_state, predictor, only(gradient))
            loss += loss_batch
        end
        loss = round(loss / length(train_loader); digits = 4)
        print("Epoch $epoch: loss = $loss\t")
        score_model(predictor, test_data)
    end
end

plots = [plot_image(predictor, test_data[i].features) for i in 1:8]
Plots.plot(plots...; size = (1200, 600), layout = (2, 4))

x, y = only(data_loader(test_data; batchsize = length(test_data)))
losses = Flux.crossentropy(predictor(x), y; agg = identity)
indices = sortperm(losses; dims = 2)[[1:4; end-3:end]]
plots = [plot_image(predictor, test_data[i].features) for i in indices]
Plots.plot(plots...; size = (1200, 600), layout = (2, 4))


function find_adversarial_image(test_case; adversary_label, δ = 0.05)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:28, 1:28] <= 1)
    @constraint(model, -δ .<= x .- test_case.features .<= δ)
    # Note: we need to use `vec` here because `x` is a 28-by-28 Matrix, but our
    # neural network expects a 28^2 length vector.
    y, _ = MathOptAI.add_predictor(model, predictor, vec(x))
    @objective(model, Max, y[adversary_label+1] - y[test_case.targets+1])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return value.(x)
end

x_adversary = find_adversarial_image(test_data[3]; adversary_label = 7);
Plots.plot(
    plot_image(predictor, test_data[3].features),
    plot_image(predictor, Float32.(x_adversary)),
)