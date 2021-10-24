module SuperNodes

using Base: @kwdef

"""
Proposed SuperArray type
Can either have 1 or 2 dimensions 

For the 1D case, for indices numbered 1...NK, I want to create a SuperArray with 2 dimensions that indexes as (N,K)
For the 2D case, for indices 1,...,(NK)^2, I want to create a SuperArray with 4 dimensions that indexes as (N,K,N,K)
"""

function expand_dims(X::Array{<:Any,1}, N::Int, K::Int)
    return reshape(X,(N,K))
end

function flatten_dims(X::Array{<:Any,2}, N::Int, K::Int)
    return reshape(X, (N*K))
end

function expand_dims(X::Array{<:Any,2}, N::Int, K::Int)
    idx_map = make_index_mapping(N,K)
	X_4D = zeros(N,K,N,K)
	for (i,(n1,k1)) in enumerate(idx_map)
		for (j,(n2,k2)) in enumerate(idx_map)
			X_4D[n1,k1,n2,k2] = X[i,j]
		end
	end
	return X_4D
end

function flatten_dims(X::Array{<:Any,4}, N::Int, K::Int)
    idx_map = make_index_mapping(N,K)
	X_2D = zeros(N*K,N*K)
	for i in 1:N*K
		for j in 1:N*K
			n_i,k_i = idx_map[i]
			n_j,k_j = idx_map[j]
			X_2D[i,j] = X[n_i,k_i,n_j,k_j]   
		end
	end
	return X_2D
end

@kwdef mutable struct SuperArray
    """
    Object that holds the data in two forms: as a 1D "flat" vector, and as a 2d "super" matrix
    Data should only be added to this struct by calling setproperty! after it is instantiated
    """
    N::Int
    K::Int
    array::Array{<:Any,1}
    matrix::Array{<:Any,2}=expand_dims(array, N, K)
end

@kwdef mutable struct SuperMatrix
    """
    Same as above, but make it a 2D "flat" matrix and a 4D "super" tensor
    """
    N::Int
    K::Int
    matrix::Array{<:Any,2}
    tensor::Array{<:Any,4}=expand_dims(matrix, N, K)
end

function Base.getindex(a::SuperArray, n::Int, k=nothing)
    if k == nothing #if we get only one index, that means we are just doing regular array indexing
        return a.array[n]
    else #return index of matrix formulation
        @assert(typeof(k)==Int)
        return a.matrix[n,k]
    end
end

function Base.getindex(a::SuperMatrix, n1::Int, k1::Int, n2=nothing, k2=nothing)
    #if we get only two indices, that means we are indexing into the flat superprocess
    if (n2 == nothing && k2 == nothing)
        return a.matrix[n1,k1]
    else
        @assert(typeof(n2)==Int && typeof(k2)==Int)
        return a.tensor[n1,k1,n2,k2]
    end
end

function Base.setindex!(a::SuperArray, v, n::Int, k=nothing)
    if k == nothing #if we get only one index, that means we are just doing regular array indexing
        a.array[n] = v 
        #but we must also update the matrix formulation
        ñ,k̃ = from_supernode(a, n) 
        a.matrix[ñ,k̃] = v
    else #we are indexing into the matrix 
        @assert(typeof(k)==Int)
        a.matrix[n,k] = v 
        #but we must also update the array formulation
        ñk̃ =  to_supernode(a, n, k)
        a.array[ñk̃] = v
    end
end

function Base.setindex!(a::SuperMatrix, v, n1::Int, k1::Int, n2=nothing, k2=nothing)
    #if we get only two indices, that means we are indexing into the flat superprocess
    if (n2 == nothing && k2 == nothing)
        a.matrix[n1,k1] = v
        #but we must also update the tensor formulation
        ñ1,k̃1,ñ2,k̃2 =  from_supernode(a, n1, k1)
        a.tensor[ñ1,k̃1,ñ2,k̃2] = v
    else #we are indexing into the tensor
        @assert(typeof(n2)==Int && typeof(k2)==Int)
        a.tensor[n1,k2,n2,k2] = v
        #but we must also update the flat matrix
        ñ1k̃1,ñ2k̃2 = to_supernode(a, n1, k1, n2, k2)
        a.matrix[ñ1k̃1,ñ2k̃2] = v
    end
end

function Base.setproperty!(a::SuperArray, name::Symbol, x)
    if name == :array
        @assert(ndims(x) == 1)
        setfield!(a, :array, x)
        setfield!(a, :matrix, expand_dims(x, a.N, a.K))
    elseif name == :matrix
        @assert(ndims(x) == 2)
        setfield!(a, :matrix, x)
        setfield!(a, :array, flatten_dims(x, a.N, a.K))
    else
        error("Invalid field")
    end
end

function Base.setproperty!(a::SuperMatrix, name::Symbol, x)
    if name == :matrix
        @assert(ndims(x) == 2)
        setfield!(a, :matrix, x)
        setfield!(a, :tensor, expand_dims(x, a.N, a.K))
    elseif name == :tensor
        @assert(ndims(x)==4)
        setfield!(a, :tensor, x)
        setfield!(a, matrix, flatten_dims(x, a.N, a.K))
    end
end

Base.copy(a::SuperArray) = SuperArray(a.N,a.K,a.array,a.matrix)

Base.copy(a::SuperMatrix) = SuperMatrix(a.N,a.K,a.matrix,a.tensor)

#size(a::SuperArray): ((a.N,a.K), a.N*a.K)

#size(a::SuperMatrix): ((a.N,a.K,a.N,a.K), (a.N*a.K)^2)

function ndims(x::AbstractArray)
    return length(size(x))
end

function to_supernode(a::SuperArray, n::Int, k::Int)
    return n+(k-1)*a.N
end

function to_supernode(a::SuperMatrix, n1::Int, k1::Int, n2::Int, k2::Int)
    return (n1+(k1-1)*a.N, n2+(k2-1)*a.N)
end

function from_supernode(a::SuperArray, nk::Int)
    k = div(nk-1,a.N)+1
    n = rem(nk,a.N)
    if n == 0
        n = a.N
    end
    return (n,k)
end

function from_supernode(a::SuperMatrix, n1k1::Int, n2k2::Int)
    n1 = rem(n1k1,a.N)
    k1 = div(n1k1-1,a.N)+1
    if n1 == 0
        n1 = a.N
    end
    n2 = rem(n2k2,a.N)
    k2 = div(n2k2-1,a.N)+1
    if n2 == 0
        n2 = a.N
    end
    return (n1,k1,n2,k2)
end

function to_supernode(N::Int, n::Int, k::Int)
    return n+(k-1)*N
end

function to_supernode(N::Int, n1::Int, k1::Int, n2::Int, k2::Int)
    return (n1+(k1-1)*N, n2+(k2-1)*N)
end

function from_supernode(N::Int, nk::Int)
    k = div(nk-1,N)+1
    n = rem(nk,N)
    if n == 0
        n = N
    end
    return (n,k)
end

function from_supernode(N::Int, n1k1::Int, n2k2::Int)
    n1 = rem(n1k1,N)
    k1 = div(n1k1-1,N)+1
    if n1 == 0
        n1 = N
    end
    n2 = rem(n2k2,N)
    k2 = div(n2k2-1,N)+1
    if n2 == 0
        n2 = N
    end
    return (n1,k1,n2,k2)
end

function make_index_mapping(N,K)
    idx_mapping = []
    for k in 1:K
        for n in 1:N
            push!(idx_mapping,(n,k))
        end
    end
    return idx_mapping
end

export SuperArray, SuperMatrix

end
