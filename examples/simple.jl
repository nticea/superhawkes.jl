using SuperHawkes
using Plots
using Profile
using Random

function accuracy(ŷ,y)
    sum(y .== ŷ) / length(y)
end

# Parameters
Profile.clear()
Random.seed!(12345)
N, T, K = 20, 100, 3
niter = 25

# Create a SuperHawkesProcess and sample spikes 
SHP = SuperHawkesProcess(N=N,T=T,K=K)
true_spikes = sample(SHP)
true_parents = get_parents(true_spikes)
true_sequenceIDs = get_sequenceIDs(true_spikes)
S = length(true_spikes)

### Initialize spikes and parents for inference
spikes = copy(true_spikes)
# Assign all spikes to sequence 1
spikes.sequenceIDs = ones(Int64,S) #this also automatically updates spikes.supernodes 
# Assign all spikes to background process
spikes.parents = rand(1:K,S)#zeros(Int64,S)
lookbacks = get_lookback_spikes(SHP.ΔT_max, spikes)

parent_acc = []
sequence_acc = []

for i in 1:niter   
    push!(parent_acc, accuracy(get_parents(spikes), true_parents))
    push!(sequence_acc, accuracy(get_sequenceIDs(spikes),true_sequenceIDs))     
    #resample parents
    sample_parents!(SHP, spikes, lookbacks)
    #spikes.parents = true_parents

    #resample sequence IDs
    alphas = forward_pass_tree(SHP, spikes)
    backward_sample_tree!(SHP, spikes, alphas, true_sequenceIDs)

    #resample posteriors
    update_posteriors!(SHP, spikes)
end

plot(1:length(sequence_acc),sequence_acc,label="Sequence assignemnts")
plot!(1:length(parent_acc),parent_acc,label="Parent assignments")

