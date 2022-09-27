module Kmeans1D

include("_core_trans.jl")

using ._Kmeans1D

function cluster_julia(array::Array{Float32}, k::Int)
    """
    :param array: A sequence of floats
    :param k: Number of clusters (int)
    :return: A tuple with (clusters, centroids)
    """
    @assert k>0
    n = length(array)
    @assert n>0
    k = min(k, n)

    centroids, clusters = _Kmeans1D.cluster(array, n, k)
    return (; clusters, centroids)
end


end #end of module

