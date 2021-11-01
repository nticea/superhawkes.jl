### MAKE INTERESTING PRIORS ###
NETWORK_SPARSITY = 0.001

## Priors on λ0

# α0 
function make_α0_prior(N::Int, K::Int)
    prior_α0 = zeros(N*K) .+ eps()
    # Each sequence sources a non-overlapping subset of nodes
    for k in 1:K
        start_idx = (k-1)*(N+1) + 1 + (k-1)*div(N,K)
        end_idx = min(N*K,start_idx+1)#div(N,K))
        prior_α0[start_idx:end_idx] .= 1
    end
    return prior_α0
end

function make_θ0_prior(N::Int, K::Int)
    prior_θ0 = repeat(rand(K),inner=N)
end

function make_αW_prior(N::Int, K::Int)
    # Within a sequence, we only have upper and lower diagonal elements 
    prior_αW = ones((N,K,N,K)) ./ (N*K)#eps() .* ones((N,K,N,K))
    @assert N>=K
    Δ = floor(Int64, N/K)
    for k in 1:K
        # Non-overlapping sections are filled 
        prior_αW[(k-1)*Δ+1:min(k*Δ+1,N),k,:,k] .= 1 
        # Add a few instances of cross-sequence spike induction
        #prior_αW[rand(1:N*K),k,rand(1:N*K),k+1] .= 1
    end
    prior_αW = flatten_dims(prior_αW,N,K)
end

function make_θW_prior(N::Int, K::Int)
    prior_θW = eps().*ones((N,K,N,K))
    for k in 1:K
        prior_θW[:,k,:,k] .= N*NETWORK_SPARSITY
    end
    prior_θW = flatten_dims(prior_θW,N,K)
end

function make_αR_prior(N::Int, K::Int)
    prior_αR = ones(N*K) 
end

function make_θR_prior(N::Int, K::Int)
    prior_θR = 1 ./ repeat(rand(K),inner=N)
end

### ACCURACY FUNCTIONS ###

function accuracy(ŷ::Vector{<:Real},y::Vector{<:Real})
    sum(y .== ŷ) / length(y)
end

function posterior_accuracy(P̂::SuperHawkesProcess, P::SuperHawkesProcess)
    # Compare the bias parameters
    Δα0 = norm(P̂.bias.α0.array - P.bias.α0.array)/norm(P.bias.α0.array)
    Δθ0 = norm(P̂.bias.θ0.array - P.bias.θ0.array)/norm(P.bias.θ0.array)
    Δλ0 = norm(P̂.bias.λ0.array - P.bias.λ0.array)#/norm(P.bias.λ0.array)

    # Compare the network parameters
    ΔαW = norm(P̂.network.αW.matrix - P.network.αW.matrix)/norm(P.network.αW.matrix)
    ΔθW = norm(P̂.network.θW.matrix - P.network.θW.matrix)/norm(P.network.θW.matrix)
    ΔW = norm(P̂.network.W.matrix - P.network.W.matrix)#/norm(P.network.W.matrix)

    # Compare the kernel parameters
    ΔαR = norm(P̂.kernel.αR.array - P.kernel.αR.array)/norm(P.kernel.αR.array)
    ΔθR = norm(P̂.kernel.θR.array - P.kernel.θR.array)/norm(P.kernel.θR.array)
    Δrate = norm(P̂.kernel.rate.array - P.kernel.rate.array)#/norm(P.kernel.rate.array)

    #return [Δα0,Δθ0,ΔαW,ΔθW,ΔαR,ΔθR]
    return [Δλ0,ΔW,Δrate]
end