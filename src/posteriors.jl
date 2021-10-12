function update_α0(α0::SuperArray, source_spikes::Array{Tuple{Float64, Int64},1}, NK::Int)
    source_nodes = [s[2] for s in source_spikes]
    h = fit(Histogram, source_nodes, 1:NK+1)
    return α0.array + h.weights
end

function update_β0(β0::SuperArray, T::Real)
    return β0.array .+ T
end

function update_π0(π0::SuperArray, source_spikes::Array{Tuple{Float64, Int64},1}, NK::Int)
    source_nodes = [s[2] for s in source_spikes]
    h = fit(Histogram, source_nodes, 1:NK+1)
    return π0.array + h.weights
end

function update_αW(αW::SuperMatrix, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    αW_new = copy(αW.matrix)
    for (s,(_,nc)) in enumerate(spike_list)
        p = parents[s]
        if p>0
            np = spike_list[p][2]
            αW_new[nc,np] += 1
        end
    end
    return αW_new
end

function update_βW(βW::SuperMatrix, spike_list::Array{Tuple{Float64, Int64},1})
    βW_new = copy(βW.matrix)
    for (_,np) in spike_list
        βW_new[:,np] .+= 1
    end
    return βW_new
end

function update_αR(αR::SuperArray, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1}, NK::Int)
    parent_nodes = [spike_list[p][2] for p in findall(x -> x>0, parents)]
    h = fit(Histogram, parent_nodes, 1:NK+1) #count the number of times a given node is a parent
    return αR.array + h.weights
end

function update_βR(βR::SuperArray, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    βR_new = copy(βR.array)
    for (s,(tc,nc)) in enumerate(spike_list)
        p = parents[s]
        if p > 0
            tp,np = spike_list[p]
            βR_new[np] += tc-tp
        end
    end
    return βR_new
end


function update_posteriors!(P::SuperHawkesProcess, spikes::Spikes)
    parents = spikes.parents
    spike_list = partially_observed_spikes(spikes)
    N = num_nodes(P)
    K = num_sequences(P)
    T = max_time(P)
    
    ## BIAS UPDATES
    source_spikes = spike_list[findall(p -> p==0, parents)]
    # Update α0
    α0_new = update_α0(P.bias.α0, source_spikes, N*K)
    # Update β0
    β0_new = update_β0(P.bias.α0, T)
    # Update π0
    π0_new = update_π0(P.bias.α0, source_spikes, N*K)

    ## NETWORK UPDATES
    # Update αW
    αW_new = update_αW(P.network.αW, spike_list, parents)
    # Update βW
    βW_new = update_βW(P.network.βW, spike_list)

    ## KERNEL UPDATES
    # Update αR
    αR_new = update_αR(P.kernel.αR, spike_list, parents, N*K)
    # Update βR
    βR_new = update_βR(P.kernel.βR, spike_list, parents)

    ## PROCESS UPDATE 
    # Create a new SuperHawkesProcess with these new attributes
    P = SuperHawkesProcess(N = N, K = K, T = T,
    prior_α0 = α0_new, prior_β0 = β0_new, prior_π0 = π0_new,
    prior_αW = αW_new, prior_βW = βW_new,
    prior_αR = αR_new, prior_βR = βR_new)
end