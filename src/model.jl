MAX_DEPTH = 4
NETWORK_SPARSITY = 0.0007
import Base: length

## PARAMETER STRUCTS ##

function make_α0(N::Int, K::Int, prior_α0=nothing)
    if isnothing(prior_α0)
        #The default is to return all ones
        prior_α0 = ones(N*K)
    end
    SuperArray(N=N,K=K,array=prior_α0)
end

function make_β0(N::Int, K::Int, prior_β0=nothing)
    if isnothing(prior_β0)
        #The default is to assign a different β0 to each sequence
        prior_β0 = repeat(rand(K),inner=N)
    end
    SuperArray(N=N,K=K,array=prior_β0)
end

function make_π0(N::Int, K::Int, prior_π0=nothing)
    if isnothing(prior_π0)
        #The default is to recruit only a subset of nodes for background on each process
        #prior_π0 = repeat(rand(K),inner=N)
        prior_π0 = eps() .* ones(N*K)
        for k in 1:K
            start_idx = (k-1)*(N+1) + 1 + (k-1)*div(N,K)
            end_idx = min(N*K,start_idx+div(N,K))
            prior_π0[start_idx:end_idx] .= 1
        end
    end
    SuperArray(N=N,K=K,array=prior_π0)
end

function sample_λ0(N::Int, K::Int, α0::SuperArray, β0::SuperArray, π0::SuperArray)
    #Create the Gammas and Dirichlet distributions
    γ0 = Gamma.(α0.array, β0.array) #Gamma with NK elements
    dirichlet = (Dirichlet(π0.array)) #Dirichlet with NK elements
    #Sample
    SuperArray(N=N,K=K,array=rand.(γ0) .* rand(dirichlet) .+ eps())
end

mutable struct Bias 
    #hypers 
    N::Int
    K::Int    
    #parameters
    α0::SuperArray
    β0::SuperArray
    π0::SuperArray
    #actual rate (sampled)
    λ0::SuperArray 
end

function Bias(N::Int,K::Int, 
              prior_α0::Union{Array{Float64,1},Nothing}=nothing,
              prior_β0::Union{Array{Float64,1},Nothing}=nothing,
              prior_π0::Union{Array{Float64,1},Nothing}=nothing)
    
    #parameters
    α0::SuperArray = make_α0(N,K,prior_α0) #NK
    β0::SuperArray = make_β0(N,K,prior_β0) #NK
    π0::SuperArray = make_π0(N,K,prior_π0) #NK

    #sample the parameters to get the bias term
    λ0::SuperArray = sample_λ0(N,K,α0,β0,π0) #NK 
    return Bias(N,K,α0,β0,π0,λ0)
end

function resample(bias::Bias)
    bias.λ0 = sample_λ0(bias.N,bias.K,bias.α0,bias.β0,bias.π0)
end

function make_αW(N::Int, K::Int, prior_αW=nothing)
    if isnothing(prior_αW)
        #low connectivity within a sequences, effectively zero connectivity between sequences 
        prior_αW = eps().*ones((N,K,N,K))
        for k in 1:K
            prior_αW[:,k,:,k] .= 1
        end
        prior_αW = flatten_dims(prior_αW,N,K)
    end
    SuperMatrix(N=N,K=K,matrix=prior_αW)
end

function make_βW(N::Int, K::Int, prior_βW=nothing)
    if isnothing(prior_βW)
        prior_βW = eps().*ones((N,K,N,K))
        for k in 1:K
            prior_βW[:,k,:,k] .= N*NETWORK_SPARSITY
        end
        prior_βW = flatten_dims(prior_βW,N,K)
        #TODO: add a few instances of cross-sequence spike induction!
    end
    SuperMatrix(N=N,K=K,matrix=prior_βW)
end

function sample_W(N::Int, K::Int, αW::SuperMatrix, βW::SuperMatrix)
    #Create distribution over the coupling matrix 
    γW = Gamma.(αW.matrix, βW.matrix)
    #Sample
    SuperMatrix(N=N,K=K,matrix=rand.(γW).+eps())
end

mutable struct Network
    #hypers 
    N::Int
    K::Int  
    #parameters
    αW::SuperMatrix
    βW::SuperMatrix
    #coupling matrix (sampled)
    W::SuperMatrix
end

function Network(N::Int,K::Int,
                prior_αW::Union{Array{Float64,2},Nothing}=nothing,
                prior_βW::Union{Array{Float64,2},Nothing}=nothing)

    #parameters
    αW::SuperMatrix = make_αW(N,K,prior_αW)
    βW::SuperMatrix = make_βW(N,K,prior_βW)

    #sample the parameters to get the coupling matrix
    W::SuperMatrix = sample_W(N,K,αW,βW)

    return Network(N,K,αW,βW,W)
end

function resample(network::Network)
    bias.W = sample_W(network.N,network.K,network.αW,network.βW)
end

function make_αR(N::Int, K::Int, prior_αR=nothing)
    if isnothing(prior_αR)
        #The default is to return all ones
        prior_αR = ones(N*K)
    end
    SuperArray(N=N,K=K,array=prior_αR)
end

function make_βR(N::Int, K::Int, prior_βR=nothing)
    if isnothing(prior_βR)
        #The default is to assign a different β0 to each sequence
        prior_βR = repeat(rand(K),inner=N)
    end
    SuperArray(N=N,K=K,array=prior_βR)
end

function sample_rate(N::Int, K::Int, αR::SuperArray, βR::SuperArray)
    #Create distribution over the rates
    γR = Gamma.(αR.array, βR.array)
    SuperArray(N=N,K=K,array=rand.(γR))
end

mutable struct Kernel 
    #hypers 
    N::Int
    K::Int  
    #parameters
    αR::SuperArray
    βR::SuperArray
    #rate (sampled)
    rate::SuperArray
end

function Kernel(N::Int,K::Int,
                prior_αR::Union{Array{Float64,1},Nothing}=nothing,
                prior_βR::Union{Array{Float64,1},Nothing}=nothing)
    #parameters
    αR::SuperArray = make_αR(N,K,prior_αR)
    βR::SuperArray = make_αR(N,K,prior_βR)

    #sample the parameters to get the kernel
    rate::SuperArray = sample_rate(N,K,αR,βR)

    return Kernel(N,K,αR,βR,rate)
end

function resample(kernel::Kernel)
    bias.rate = sample_rate(kernel.N,kernel.K,kernel.αR,kernel.βR)
end

#rewrite the pdf method for the Kernel struct
function evaluate_pdf(kernel::Kernel, ΔT::Real, n::Int, k=nothing)
    if isnothing(k) #if we get only one index, that means we are indexing into the flattened superprocess
        return pdf(Exponential(1/kernel.rate[n]), ΔT)
    else #return index of matrix formulation
        @assert(typeof(k)==Int)
        return pdf(Exponential(1/kernel.rate[n,k]), ΔT)
    end
end

function sample(kernel::Kernel, n::Int, k=nothing)
    if isnothing(k) #if we get only one index, that means we are indexing into the flattened superprocess
        return rand(Exponential(1/kernel.rate[n]))
    else #return index of matrix formulation
        @assert(typeof(k)==Int)
        return rand(Exponential(1/kernel.rate[n,k]))
    end
end

## SUPERHAWKES STRUCT ##

mutable struct SuperHawkesProcess
    N::Int
    K::Int
    T::Real
    ΔT_max::Real #maximum temporal separation between events

    #the following hyperparameters are for the "superprocess" -- the flattened Hawkes process
    bias::Bias
    network::Network
    kernel::Kernel
end

function SuperHawkesProcess(;N::Int,K::Int=1,T::Real, #unless specified, we assume only 1 sequence type
                            prior_α0::Union{Array{Float64,1},Nothing}=nothing,
                            prior_β0::Union{Array{Float64,1},Nothing}=nothing,
                            prior_π0::Union{Array{Float64,1},Nothing}=nothing,
                            prior_αW::Union{Array{Float64,2},Nothing}=nothing,
                            prior_βW::Union{Array{Float64,2},Nothing} = nothing,
                            prior_αR::Union{Array{Float64,1},Nothing} = nothing,
                            prior_βR::Union{Array{Float64,1},Nothing} = nothing)
    
    #Check the dimensions of the inputs
    @assert(isnothing(prior_α0) || size(prior_α0)==(N*K,))
    @assert(isnothing(prior_β0) || size(prior_β0)==(N*K,))
    @assert(isnothing(prior_π0) || size(prior_π0)==(N*K,))
    @assert(isnothing(prior_αW) || size(prior_αW)==(N*K,N*K))
    @assert(isnothing(prior_βW) || size(prior_βW)==(N*K,N*K))
    @assert(isnothing(prior_αR) || size(prior_αR)==(N*K,))
    @assert(isnothing(prior_βR) || size(prior_βR)==(N*K,))

    #compute the relevant attributes of the Hawkes struct
    ΔT_max = T/10 
    bias = Bias(N,K,prior_α0,prior_β0,prior_π0)
    network = Network(N,K,prior_αW,prior_βW)
    kernel = Kernel(N,K,prior_αR,prior_βR)
    
    return SuperHawkesProcess(N,K,T,ΔT_max,bias,network,kernel)
end

num_nodes(P::SuperHawkesProcess) = P.N
num_sequences(P::SuperHawkesProcess) = P.K
num_supernodes(P::SuperHawkesProcess) = P.N*P.K
max_time(P::SuperHawkesProcess) = P.T

function check_W(P::SuperHawkesProcess)
    W = P.network.W.matrix
    branching_ratios = sum(W,dims=2)
    return !any(x->x==3, branching_ratios)
end

## STRUCT FOR HOLDING SPIKE INFORMATION ##

mutable struct Spikes
    N::Int
    K::Int
    times::Array{Real,1} # immutable
    supernodes::Array{Int,1} # must update sequenceIDs whenever this is updated
    actualnodes::Array{Int,1} # immutable
    sequenceIDs::Array{Int,1} # must update supernodes whenever this is updated
    parents::Array{Int,1} 
end

"""
Whenever we update supernodes or sequenceIDs, we must update the other
"""
function Base.setproperty!(a::Spikes, name::Symbol, x)
    if name == :supernodes
        @assert(length(x)==length(a))
        setfield!(a, :supernodes, x)
        nodes = from_supernode.(a.N,x)
        actualnodes = [n[1] for n in nodes]
        sequenceIDs = [n[2] for n in nodes]
        @assert(actualnodes == a.actualnodes)
        setfield!(a, :sequenceIDs, sequenceIDs)
    elseif name == :sequenceIDs
        @assert(length(x) == length(a))
        setfield!(a, :sequenceIDs, x)
        setfield!(a, :supernodes, to_supernode.(a.N,a.actualnodes,x))
    else
        setfield!(a, name, x)
    end
end

Base.copy(a::Spikes) = Spikes(a.N,a.K,a.times,a.supernodes,a.actualnodes,a.sequenceIDs,a.parents)

"""
Only returns information about which supernode the spike occurred on
"""
function partially_observed_spikes(spikes::Spikes)
    t, sn = spikes.times, spikes.supernodes
    @assert(length(t) == length(sn))
    [(t[s], sn[s]) for s in 1:length(t)]
end

"""
Returns information about the actual node and sequence type the spike occurred on
"""
function fully_observed_spikes(spikes::Spikes)
    t, sn, an, IDs = spikes.times, spikes.supernodes, spikes.actualnodes, spikes.sequenceIDs
    @assert(length(t) == length(sn) == length(an) == length(IDs))
    [(t[s], an[s], IDs[s]) for s in 1:length(t)]
end

"""
Returns # of spikes 
"""
length(spikes::Spikes) = length(spikes.times)
get_parents(spikes::Spikes) = spikes.parents
get_sequenceIDs(spikes::Spikes) = spikes.sequenceIDs

## METHODS FOR SUPERHAWKES PROCESS ## 

function sample(P::SuperHawkesProcess)
    #before sampling, check that the generative process is finite
    @assert(check_W(P))

    spikes = []
    parents = []

    for s in sample_base_spikes(P)
        push!(spikes, s)
        push!(parents, 0)

        ωs = length(parents)
        generate_descendants!(P, spikes, parents, s, ωs)
    end

    order = sortperm(spikes, by=s -> s[2])
    spikes = spikes[order]
    parents = parents[order]
    parents = [ω == 0 ? 0 : findfirst(==(ω), order) for ω in parents]

    #repackage spikes using the struct defined earlier
    supernodes = [s[1] for s in spikes]
    times = [s[2] for s in spikes]
    actualnodes = [s[3] for s in spikes]
    sequenceIDs = [s[4] for s in spikes]

    return Spikes(P.N,P.K,times,supernodes,actualnodes,sequenceIDs,parents)
end

function sample_base_spikes(P::SuperHawkesProcess)
    NK, T = num_supernodes(P), max_time(P)

    base_spikes = []
    for n in 1:NK
        Sn = rand(Poisson(P.bias.λ0[n] * T))
        ñ,k̃ = from_supernode(P.bias.λ0,n) #get the actual node and process indices
        append!(base_spikes, [(n, rand(Uniform(0, T)), ñ, k̃) for s in 1:Sn])
    end
    return base_spikes
end

function generate_descendants!(P::SuperHawkesProcess, spikes, parents, s, ωs, depth=0)
    NK, T  = num_supernodes(P), max_time(P)
    (n, t, ñ, k̃) = s

    if depth > MAX_DEPTH
        @warn "Max depth exceeded"
        return
    end

    for nc in 1:NK
        S_nc = rand(Poisson(P.network.W[nc, n]))
        ñc̃, k̃c̃, ñp̃, k̃p̃ = from_supernode(P.network.W, nc, n)
        @assert(ñp̃ == ñ && k̃p̃ == k̃)

        for _ in 1:S_nc
            tc = t + sample(P.kernel, n)
            if tc <= T
                sc = (nc, tc, ñc̃, k̃c̃)

                push!(spikes, sc)
                push!(parents, ωs)

                ωc = length(parents)
                generate_descendants!(P, spikes, parents, sc, ωc, depth+1)
            end
        end
    end
end

#TODO: Create a plotting function! 
function plot_spikes(spikes::Spikes)
    scatter(spikes.times, spikes.actualnodes, c=spikes.sequenceIDs)
end