using BenchmarkTools

include("scr/core.jl")
using .Kmeans1D

X = rand(Float32, 1000)
k = 32
@btime Kmeans1D.cluster_julia(X,k)