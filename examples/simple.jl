using SuperHawkes
using Plots
using Profile
using Random
using LinearAlgebra

function accuracy(ŷ,y)
    sum(y .== ŷ) / length(y)
end

function posterior_accuracy(P̂::SuperHawkesProcess, P::SuperHawkesProcess)
    # Compare the bias parameters
    Δα0 = norm(P̂.bias.α0.array - P.bias.α0.array)
    Δθ0 = norm(P̂.bias.θ0.array - P.bias.θ0.array)
    Δλ0 = norm(P̂.bias.λ0.array - P.bias.λ0.array)

    # Compare the network parameters
    ΔαW = norm(P̂.network.αW.matrix - P.network.αW.matrix)
    ΔθW = norm(P̂.network.θW.matrix - P.network.θW.matrix)
    ΔW = norm(P̂.network.W.matrix - P.network.W.matrix)

    # Compare the kernel parameters
    ΔαR = norm(P̂.kernel.αR.array - P.kernel.αR.array)
    ΔθR = norm(P̂.kernel.θR.array - P.kernel.θR.array)
    Δrate = norm(P̂.kernel.rate.array - P.kernel.rate.array)

    return [Δα0,Δθ0,ΔαW,ΔθW,ΔαR,ΔθR]
    #return [Δλ0,ΔW,Δrate]
end

# Parameters
Random.seed!(1234)
Profile.clear()
N, T, K = 20, 100, 3
niter = 100

# Create a SuperHawkesProcess for sampling
true_SHP = SuperHawkesProcess(N=N,T=T,K=K)
true_spikes = sample(true_SHP)
true_parents = get_parents(true_spikes)
true_sequenceIDs = get_sequenceIDs(true_spikes)
S = length(true_spikes)

## Create a SuperHawkes Process for fitting
SHP = true_SHP#SuperHawkesProcess(N=N,T=T,K=K)

### Initialize spikes and parents for inference
spikes = copy(true_spikes)
spikes.sequenceIDs = rand(1:K,S) # Assign all spikes to sequence 1. This also automatically updates spikes.supernodes 
spikes.parents = zeros(Int64,S) # Assign all spikes to background process

parent_acc = []
sequence_acc = []
posterior_acc = []

lookbacks = get_lookback_spikes(SHP.ΔT_max, spikes)
for i in 1:niter   
    push!(parent_acc, accuracy(get_parents(spikes), true_parents))
    push!(sequence_acc, accuracy(get_sequenceIDs(spikes),true_sequenceIDs))  
    push!(posterior_acc, posterior_accuracy(SHP, true_SHP))
    
    # resample posteriors
    update_posteriors!(SHP, spikes)
    
    # resample sequence IDs. NOTE: code runs much better when this precedes the parent resampling
    alphas = forward_pass_tree(SHP, spikes)
    backward_sample_tree!(SHP, spikes, alphas)

    # resample parents
    sample_parents!(SHP, spikes, lookbacks, true_parents)
end

plot(1:niter,sequence_acc,label="Sequence assignemnts")
plot!(1:niter,parent_acc,label="Parent assignments")
#plot!(1:niter,posterior_acc,label="Posteriors")


