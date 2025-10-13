# %% read and test
using HDF5

nl_list   = [20; 38; 186]
bus_list  = [14; 24; 118]
node_list = [32; 128; 512; 2048]
pg_list   = ["case14_"; "case24_ieee"; "pglib_opf_case118_ieee.m"]

# %% Create plot!
c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

plots = []
for bus in bus_list
    for node in node_list
        data_file_ai = "data/"*string(bus)*"bus_"*string(node)*"node_MathOptAI_test.h5"
        data_file_sample = "data/"*string(bus)*"bus_"*string(node)*"node_sampling_test.h5"

        fid   = h5open(data_file_ai, "r")
        obj_soc_zrelax_ai   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_ai    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_ai  = read(fid, "obj_acopf_zsnap")
        close(fid)

        fid                     = h5open(data_file_sample, "r")
        obj_soc_zrelax_sample   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_sample    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_sample  = read(fid, "obj_acopf_zsnap")
        close(fid)

        title = string(bus)*"-bus "*string(node)*"NN"

        if node == 32
            if bus == 118
                p = scatter(obj_soc_zrelax_sample, label="random sample",legendfont = 8, ylabel="Load Shed (pu)", xlabel="Sample index", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
                plot!(p, obj_soc_zrelax_ai*ones(100), label="MathOptAI bound",legend = :topright, foreground_color_legend = nothing, width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash)
            else
                p = scatter(obj_soc_zrelax_sample, label="", ylabel="Load Shed (pu)", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
                plot!(p, obj_soc_zrelax_ai*ones(100), label="", width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash)
            end

        elseif bus == 118
            p = scatter(obj_soc_zrelax_sample, label="random sample",legendfont = 8, xlabel="Sample index", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
            plot!(p, obj_soc_zrelax_ai*ones(100), label="MathOptAI bound",legend = :topright, foreground_color_legend = nothing, width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash) 
        else
            p = scatter(obj_soc_zrelax_sample, label="", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
            plot!(p, obj_soc_zrelax_ai*ones(100), label="", width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash)
        end

        push!(plots,p)
    end
end

plot(plots..., layout=(3,4), size=(1000, 600), left_margin=3Plots.mm, bottom_margin=3.5Plots.mm)
# => savefig("bound.pdf")

# %% get Table I data
TmI   = zeros(4,6)
TmII  = zeros(4,6)
TmIII = zeros(4,6)
row = 1
for node in node_list
    col = 1
    for bus in bus_list
        data_file_ai = "data/"*string(bus)*"bus_"*string(node)*"node_MathOptAI_test.h5"
        data_file_sample = "data/"*string(bus)*"bus_"*string(node)*"node_sampling_test.h5"

        fid   = h5open(data_file_ai, "r")
        obj_soc_zrelax_ai   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_ai    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_ai  = read(fid, "obj_acopf_zsnap")
        dt                  = read(fid, "solve_time_mathoptai")
        println(dt)
        close(fid)

        println()
        fid                     = h5open(data_file_sample, "r")
        obj_soc_zrelax_sample   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_sample    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_sample  = read(fid, "obj_acopf_zsnap")
        close(fid)

        TmI[row,2*col-1] = round(obj_soc_zrelax_ai; digits=2)
        TmI[row,2*col  ] = round(maximum(obj_soc_zrelax_sample); digits=2)

        TmII[row,2*col-1] = round(obj_soc_zsnap_ai; digits=2)
        TmII[row,2*col  ] = round(maximum(obj_soc_zsnap_sample); digits=2)

        TmIII[row,2*col-1] = round(obj_acopf_zsnap_ai; digits=2)
        TmIII[row,2*col  ] = round(maximum(obj_acopf_zsnap_sample); digits=2)

        col+=1
    end
    row+=1
end