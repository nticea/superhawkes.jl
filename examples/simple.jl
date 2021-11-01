using SuperHawkes
using Plots
using Profile
using Random
using LinearAlgebra

# Parameters
Random.seed!(1234)
Profile.clear()
N, T, K = 20, 100, 3
niter = 500

α0 = make_α0_prior(N,K)
θ0 = make_θ0_prior(N,K)
αW = make_αW_prior(N,K)
θW = make_θW_prior(N,K)
αR = make_αR_prior(N,K)
θR = make_θR_prior(N,K)

# Create a SuperHawkesProcess for sampling
true_SHP = SuperHawkesProcess(N=N,K=K,T=T,prior_α0=α0,prior_θ0=θ0,prior_αW=αW,prior_θW=θW,prior_αR=αR,prior_θR=θR)
println("Sampling from true Hawkes process")
true_spikes = sample(true_SHP)
true_parents = get_parents(true_spikes)
true_sequenceIDs = get_sequenceIDs(true_spikes)
S = length(true_spikes)

## Create a SuperHawkes Process for fitting
println("Creating a test Hawkes process using uninformative priors")
SHP = SuperHawkesProcess(N=N,T=T,K=K) #uninformative priors

### Initialize spikes and parents for inference
spikes = copy(true_spikes)
#spikes.sequenceIDs = rand(1:K,S) # Assign all spikes to random sequence. This also automatically updates spikes.supernodes 
#spikes.parents = zeros(Int64,S) # Assign all spikes to background process

parent_acc = []
nonzero_parent_acc = []
sequence_acc = []
posterior_acc = []

lookbacks = get_lookback_spikes(SHP.ΔT_max, spikes)
println("Fitting test Hawkes process to data")
for i in 1:niter
    print(i,"-")
    push!(parent_acc, accuracy(get_parents(spikes), true_parents))
    push!(nonzero_parent_acc, accuracy(get_parents(spikes)[true_parents .> 0], true_parents[true_parents .> 0]))
    push!(sequence_acc, accuracy(get_sequenceIDs(spikes),true_sequenceIDs))  
    push!(posterior_acc, posterior_accuracy(SHP, true_SHP))
    
    # resample posteriors
    update_posteriors!(SHP, spikes, true_SHP)
    
    # resample sequence IDs. NOTE: code runs much better when this precedes the parent resampling
    #alphas = forward_pass_tree(SHP, spikes)
    #backward_sample_tree!(SHP, spikes, alphas)

    # resample parents
    #sample_parents!(SHP, spikes, lookbacks, true_parents)
end

plot(1:niter,sequence_acc,label="Sequence assignemnts")
plot!(1:niter,parent_acc,label="Parent assignments")
plot!(1:niter,nonzero_parent_acc,label="Nonzero parent assignments")

plot(1:niter,[p[1] for p in posterior_acc],label="λ0 assignments")
plot!(1:niter,[p[2] for p in posterior_acc],label="W assignments")
plot!(1:niter,[p[3] for p in posterior_acc],label="rate assignments")



