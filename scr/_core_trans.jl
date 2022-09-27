module _Kmeans1D

function smawk(rows, cols, lookup)
    #Recursion base case
    if length(rows) == 0
        return Dict()
    end

    ## REDUCE
    _cols = []
    for col in cols
        while true
            if length(_cols) == 0
                break
            end
            index = length(_cols)

            if lookup[rows[index], col] >= lookup[rows[index], _cols[end]]
                break
            end
            pop!(_cols)
        end
        
        if length(_cols) < length(rows)
            push!(_cols, col)
        end
        
    end
    # call recursively on odd-indexed rows
    odd_rows = []
    if length(rows) > 1 
        for i in 2:2:length(rows)
            push!(odd_rows, rows[i])
        end
    end


    result = smawk(odd_rows, _cols, lookup)

    col_idx_lookup = Dict()
    for idx in 1:length(_cols)
        col_idx_lookup[_cols[idx]] = idx
    end


    ## INTERPOLATE

    #Fill-in even-indexed rows
    start = 1

    for r in 1:2:length(rows)
        row = rows[r]
        stop = length(_cols) - 1
        if r < (length(rows) - 1)
            stop = col_idx_lookup[result[rows[r + 1]]]
        end
        argmin = _cols[start]
        min = lookup[row, argmin]

        for c in start+1:stop+1
            value = lookup[row, _cols[c]]
            if (c == start) || (value<min)
                argmin = _cols[c]
                min = value
            end
        end
        
        result[row] = argmin
        start = stop

    end

    return result
end


mutable struct CostCalculator
    cumsum :: Array{Float32}
    cumsum2 :: Array{Float32}

    function CostCalculator()
        new([0.0], [0.0])
    end
end

function build_cumsum(cc::CostCalculator, array, n)
    for i in 1:n
        push!(cc.cumsum, array[i]+cc.cumsum[i])
        push!(cc.cumsum2, array[i]*array[i]+cc.cumsum2[i])
    end
end

function calc(cc::CostCalculator, i, j)
    if j<i
        return 0
    end
    mu = (cc.cumsum[j + 1] - cc.cumsum[i]) / (j - i + 1)
    result = cc.cumsum2[j + 1] - cc.cumsum2[i]
    result += (j - i + 1) * (mu * mu)
    result -= (2 * mu) * (cc.cumsum[j + 1] - cc.cumsum[i])

    return result
end

function C_builder(C, D, k, n, cc)
    for i in 1:n
        for j in 1:n
            col = min(i, j-1)
            if col < 1
                col = n + col
            end
            C[i, j] = D[k-1, col] + calc(cc, j, i)
        end
    end
    return C
end


function cluster(array, n, k)
    #Sort input array and save info for de-sorting
    sort_idx = sortperm(array)
    undo_sort_lookup = Vector{Int32}(undef, n)
    sorted_array = Vector{Float32}(undef, n)
    for i in 1:n
        sorted_array[i] = array[sort_idx[i]]
        undo_sort_lookup[sort_idx[i]] = i 

    end

    #Set D and T using dynamic programming algorithm
    cost_calculator = CostCalculator()
    build_cumsum(cost_calculator::CostCalculator, sorted_array, n)


    D = Matrix{Float32}(undef, k, n)
    T = Matrix{Int32}(undef, k, n)

    for i in 1:n
        D[1, i] = calc(cost_calculator , 1, i)
        T[1, i] = 1
    end
   
    C = Matrix{Float32}(undef, n, n)
    row_argmins = Dict{Int32, Int32}()
    for k_ in 2:k
        #THIS IS DONE DIFFERENTLY ON C++ IMPLEMENTATION
        C = C_builder(C, D, k_, n, cost_calculator)
        row_argmins = smawk(1:n, 1:n, C)        
        for i in 1:length(row_argmins)
            argmin = row_argmins[i]
            min = C[i, argmin]
            D[k_, i] = min
            T[k_, i] = argmin
        end

    end


    #Extract cluster assignments by backtracking
    centroids = zeros(k)
    sorted_clusters = Vector{Int32}(undef, n)
    t = n+1
    k_ = k 
    n_ = n 

    while t > 1
        t_ = t
        t = T[k_,n_]
        centroid = 0.0
        for i in t:t_-1
            sorted_clusters[i] = k_
            centroid += (sorted_array[i] - centroid) / (i - t + 1)
        end
        centroids[k_] = centroid
        k_ -= 1
        n_ = t - 1
    end
    
    clusters = Vector{Int32}(undef, n)
    for i in 1:n
        clusters[i] = sorted_clusters[undo_sort_lookup[i]]
    end

    return centroids, clusters
end

end #end of module







