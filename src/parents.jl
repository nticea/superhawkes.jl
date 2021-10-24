"""
Gibbs sampler for the parent assignments 
"""

"""
    get_lookback_spikes(τ_max::Real, spikes)

    ΔT_max::Real: maximum time difference between parent and child spike
    spikes::Array{Int,1}, length S: output of partially_observed_spikes(::Spikes)
"""
function get_lookback_spikes(τ_max::Real, spikes::Spikes) 
    spike_list = partially_observed_spikes(spikes)
    lookback_spikes = [
        #tc - ΔT_max < tp < tc
        findall(s -> t - τ_max < s[1] < t, spike_list) for (t,n) in spike_list
    ]
    return lookback_spikes  
end

"""
    sample_parents!(P::SuperHawkesProcess, spikes::Spikes, lookbacks::Array{Int,1})

    P::SuperHawkesProcess is the SHP we are trying to fit
    spikes::Spikes
    lookbacks::Array{Int,1}, length S: list of vectors containing potential parents for each spike
"""
function sample_parents!(P::SuperHawkesProcess, spikes::Spikes, lookbacks::Vector{Vector{Int64}}, true_parents)
    #extract the background and coupling matrix
    spike_list = copy(partially_observed_spikes(spikes))
    parent_list = copy(spikes.parents)

    λ0, W, kernel = P.bias.λ0, P.network.W, P.kernel

    for (j, (t, n)) in enumerate(spike_list)
        S_close = length(lookbacks[j])
        probs = zeros(S_close + 1)
        probs[1] = λ0[n]
        
        for (ωj, (tp, np)) in enumerate(spike_list[lookbacks[j]])
            probs[ωj+1] = W[n, np] * evaluate_pdf(kernel, t-tp, np)
        end
        probs /= sum(probs)
        sample = rand(Categorical(probs)) - 1
        if sample == 0
            parent_list[j] = sample
        else
            parent_list[j] = lookbacks[j][sample]
        end
        # if parent_list[j] != true_parents[j] && true_parents[j]>0
        #     tp,np = spike_list[true_parents[j]]

        #     println("")
        #     println("Spike: ", j, " True parent: ", true_parents[j])
        #     for (ωj, (tp, np)) in enumerate(spike_list[lookbacks[j]])
        #         probs[ωj+1] = W[n, np] * evaluate_pdf(kernel, t-tp, np)
        #         println("probability of ", lookbacks[j][ωj], ": ", probs[ωj+1])
        #     end

        #     #println("Probs: ", probs)
        #     println("")
        #     println("True parent network: ", W[n, np])
        #     println("True parent kernel: ", evaluate_pdf(kernel, t-tp, np))
        #     println("True parent prob: ",  W[n, np]*evaluate_pdf(kernel, t-tp, np))
        #     println("Sampled parent: ", parent_list[j])
        #     println("Sampled parent probs: ", probs[sample+1])
        #     # full_spike_list = fully_observed_spikes(spikes)
        #     # println("True parent node: ", full_spike_list[true_parents[j]][2])
        #     # println("True parent process: ", full_spike_list[true_parents[j]][3])
        #     # println("True child node: ", full_spike_list[j][2])
        #     # println("True child process: ", full_spike_list[j][3])
        #     # println("True parent network (indexing into tensor): ", W[full_spike_list[j][2],full_spike_list[j][3],
        #     #         full_spike_list[true_parents[j]][2],full_spike_list[true_parents[j]][3]])
        # end
    end
    spikes.parents = convert(Array{Int,1},parent_list)
end