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
function sample_parents!(P::SuperHawkesProcess, spikes::Spikes, lookbacks::Vector{Vector{Int64}})
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
    end
    spikes.parents = convert(Array{Int,1},parent_list)
end