function update_α0(α0::SuperArray, source_spikes::Array{Tuple{Float64, Int64},1})
    N = α0.N
    K = α0.K
    source_nodes = [s[2] for s in source_spikes]
    h = fit(Histogram, source_nodes, 1:N*K+1)
    α0_new = α0.array + h.weights
    return SuperArray(N=N, K=K, array=α0_new)
end

function update_θ0(θ0::SuperArray, T::Real)
    N = θ0.N
    K = θ0.K
    β0 = 1 ./ θ0.array
    θ0_new = 1 ./ (β0 .+= T)
    return SuperArray(N=N, K=K,array=θ0_new)
end

function update_αW(αW::SuperMatrix, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    N = αW.N
    K = αW.K
    αW_new = copy(αW.matrix)
    for (s,(_,nc)) in enumerate(spike_list)
        p = parents[s]
        if p>0
            np = spike_list[p][2]
            αW_new[nc,np] += 1
        end
    end
    return SuperMatrix(N=N, K=K, matrix=αW_new)
end

function update_θW(θW::SuperMatrix, spike_list::Array{Tuple{Float64, Int64},1})
    N = θW.N
    K = θW.K
    βW_new = 1 ./ copy(θW.matrix)
    for (_,np) in spike_list
        βW_new[:,np] .+= 1
    end
    θW_new = 1 ./ βW_new
    return SuperMatrix(N=N, K=K, matrix=θW_new)
end

function update_αR(αR::SuperArray, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1}, NK::Int)
    N = αR.N
    K = αR.K
    parent_nodes = [spike_list[p][2] for p in findall(x -> x>0, parents)]
    h = fit(Histogram, parent_nodes, 1:NK+1) #count the number of times a given node is a parent
    αR_new = αR.array + h.weights
    return SuperArray(N=N, K=K, array = αR_new)
end

function update_θR(θR::SuperArray, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    N = θR.N
    K = θR.K
    βR_new = 1 ./ copy(θR.array)
    for (s,(tc,nc)) in enumerate(spike_list)
        p = parents[s]
        if p > 0
            tp,np = spike_list[p]
            βR_new[np] += tc-tp
        end
    end
    θR_new = 1 ./ βR_new
    return SuperArray(N=N, K=K, array=θR_new)
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
    α0_new = update_α0(P.bias.α0, source_spikes)
    # Update θ0
    θ0_new = update_θ0(P.bias.θ0, T)

    ## NETWORK UPDATES
    # Update αW
    αW_new = update_αW(P.network.αW, spike_list, parents)
    # Update θW
    θW_new = update_θW(P.network.θW, spike_list)

    ## KERNEL UPDATES
    # Update αR
    αR_new = update_αR(P.kernel.αR, spike_list, parents, N*K)
    # Update θR
    θR_new = update_θR(P.kernel.θR, spike_list, parents)

    ## PROCESS UPDATE 
    P.bias.λ0 = sample_λ0(N, K, α0_new, θ0_new)
    P.network.W = sample_W(N, K, αW_new, θW_new)
    P.kernel.rate = sample_rate(N, K, αR_new, θR_new)



    # Create a new SuperHawkesProcess with these new attributes
    # P_new = SuperHawkesProcess(N = N, K = K, T = T,
    # prior_α0 = α0_new, prior_θ0 = θ0_new, 
    # prior_αW = αW_new, prior_θW = θW_new,
    # prior_αR = αR_new, prior_θR = θR_new)

    # # Update the old SuperHawkesProcess
    # ## NOTE: I should've just been able to redefine P = P_new... why doesn't this work?
    # P.bias.α0 = P_new.bias.α0
    # P.bias.θ0 = P_new.bias.θ0
    # P.bias.λ0 = P_new.bias.λ0
    # P.network.αW = P_new.network.αW
    # P.network.θW =  P_new.network.θW 
    # P.network.W = P_new.network.W
    # P.kernel.αR = P_new.kernel.αR 
    # P.kernel.θR =  P_new.kernel.θR
    # P.kernel.rate = P_new.kernel.rate 
end