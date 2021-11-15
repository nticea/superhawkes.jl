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
priors = "perturbed" #"true" #"uninformative"#"perturbed"#"copy"
η = 0.1# Strength of perturbation
fix_parents = false
fix_parameters = false
fix_sequence_types = false

α0 = make_α0_prior(N,K)
θ0 = make_θ0_prior(N,K)
αW = make_αW_prior(N,K)
θW = make_θW_prior(N,K)
αR = make_αR_prior(N,K)
θR = make_θR_prior(N,K)

# Create a SuperHawkesProcess for sampling
true_SHP = SuperHawkesProcess(N=N,K=K,T=T,prior_α0=α0,prior_θ0=θ0,prior_αW=αW,prior_θW=θW,prior_αR=αR,prior_θR=θR)

# Sample
println("Sampling from true Hawkes process with ", K, " sequences")
true_spikes = sample(true_SHP)
true_parents = get_parents(true_spikes)
true_sequenceIDs = get_sequenceIDs(true_spikes)

## Create a SuperHawkes Process for fitting
if priors == "uninformative"
    println("Creating a test Hawkes drawn from uninformative priors")
    SHP = SuperHawkesProcess(N=N,T=T,K=K) #uninformative priors
elseif priors == "true"
    println("Creating a test Hawkes process drawn from the true priors")
    SHP = SuperHawkesProcess(N=N,K=K,T=T,prior_α0=α0,prior_θ0=θ0,prior_αW=αW,prior_θW=θW,prior_αR=αR,prior_θR=θR)
elseif priors == "copy"
    println("Copying the generative model")
    SHP = copy(true_SHP)
elseif priors == "perturbed"
    α0 = make_α0_prior_flat(N,K,η)
    θ0 = make_θ0_prior_flat(N,K)
    αW = make_αW_prior_flat(N,K,η)
    θW = make_θW_prior_flat(N,K)
    αR = make_αR_prior_flat(N,K)
    θR = make_θR_prior_flat(N,K)
    SHP = SuperHawkesProcess(N=N,K=K,T=T,prior_α0=α0,prior_θ0=θ0,prior_αW=αW,prior_θW=θW,prior_αR=αR,prior_θR=θR)
end

## Initialize spikes and parents for inference
S = length(true_spikes)
spikes = copy(true_spikes)
if ! fix_parents
    spikes.parents = zeros(Int64,S) # Assign all spikes to background process
else
    println("Fixing parents to true values")
end
if ! fix_sequence_types
    spikes.sequenceIDs = rand(1:K,S) # This also automatically updates spikes.supernodes 
else
    println("Fixing sequence types to true values")
end

## For tracking training
parent_acc = []
nonzero_parent_acc = []
sequence_acc = []
posterior_acc = []
logjoint_prob = []
loglike_prob = []
exact_loglike_prob = []

## Fit the model
println("Fitting test Hawkes process to data")
lookbacks = get_lookback_spikes(SHP.ΔT_max, spikes)
for i in 1:niter
    print(i,"-")
    push!(parent_acc, accuracy(get_parents(spikes), true_parents))
    push!(nonzero_parent_acc, accuracy(get_parents(spikes)[true_parents .> 0], true_parents[true_parents .> 0]))
    push!(sequence_acc, accuracy(get_sequenceIDs(spikes),true_sequenceIDs))  
    push!(posterior_acc, posterior_accuracy(SHP, true_SHP))
    push!(logjoint_prob, logjoint(SHP, spikes))
    push!(loglike_prob, loglike_data(SHP, spikes))
    if K == 1
        push!(exact_loglike_prob, exact_loglike(SHP, spikes))
    end
    
    # resample sequence IDs. NOTE: code runs much better when this precedes the parent resampling
    if ! fix_sequence_types
        alphas = forward_pass_tree(SHP, spikes)
        backward_sample_tree!(SHP, spikes, alphas)
    end

    # resample parents
    if ! fix_parents
        sample_parents!(SHP, spikes, lookbacks, true_parents)
    end

    # resample posteriors
    if ! fix_parameters
        update_posteriors!(SHP, spikes, true_SHP)
    end
end


plot(1:niter,sequence_acc,label="Sequence assignemnts")
plot!(1:niter,parent_acc,label="Parent assignments")
plot!(1:niter,nonzero_parent_acc,label="Nonzero parent assignments")

# plot(1:niter,[p[1] for p in posterior_acc],label="λ0 MSE")
# plot!(1:niter,[p[2] for p in posterior_acc],label="W MSE")
# plot!(1:niter,[p[3] for p in posterior_acc],label="rate MSE")

# display(plot(1:niter, loglike_prob,label="loglike"))
# display(plot(1:niter, logjoint_prob,label="logjoint"))

# if K==1 
#     plot(1:niter, exact_loglike_prob,label="Loglike for K=1")
#     plot!(1:niter, loglike_prob,label="Loglike with parent augmentation for K=1")
#     hline!([exact_loglike(true_SHP, true_spikes)], linestyle=:dash, label="True loglike")
# end

# plot(1:niter,[p[1] for p in logprob_priors],label="λ0 logprob")
# plot!(1:niter,[p[2] for p in logprob_priors],label="W logprob")
# plot!(1:niter,[p[3] for p in logprob_priors],label="rate logprob")

if priors == "perturbed"
    title!("Perturbation = ", η)
end



