module TopSwitchTests
    import LinearSOC
    import PGLib, Random
    import Gurobi, Ipopt
    import Printf
    import HDF5

    include("create_datafile.jl")
    
    export generate_pd_qd!

end
