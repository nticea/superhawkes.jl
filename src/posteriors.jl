function update_α0(α0::SuperArray, background_spikes::Array{Tuple{Float64, Int64},1})
    N = α0.N
    K = α0.K
    background_nodes = [s[2] for s in background_spikes]
    h = fit(Histogram, background_nodes, 1:N*K+1) # Count the number of background spikes on each node
    α0_new = α0.array + h.weights
    return SuperArray(N=N, K=K, array=α0_new)
end

function update_θ0(θ0::SuperArray, T::Real)
    N = θ0.N
    K = θ0.K
    β0_new = 1 ./ copy(θ0.array)
    θ0_new = 1 ./ (β0_new .+= T) #Increment by T
    return SuperArray(N=N,K=K,array=θ0_new)
end

function update_αW(αW::SuperMatrix, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    N = αW.N
    K = αW.K
    αW_new = copy(αW.matrix)
    for (sc, (tc,nc)) in enumerate(spike_list)
        for (sp, (tp,np)) in enumerate(spike_list)
            if parents[sc] == sp
                αW_new[nc,np] += 1
            end
        end
    end

    # for (s,(_,nc)) in enumerate(spike_list)
    #     p = parents[s]
    #     if p>0
    #         np = spike_list[p][2]
    #         αW_new[nc,np] += 1
    #     end
    # end
    return SuperMatrix(N=N, K=K, matrix=αW_new)
end

function update_θW(θW::SuperMatrix, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    N = θW.N
    K = θW.K
    βW_new = 1 ./ copy(θW.matrix)
    for (_,np) in spike_list
        βW_new[:,np] .+= 1
    end
    # for (s,(_,nc)) in enumerate(spike_list)
    #     p = parents[s]
    #     if p>0
    #         tp,np = spike_list[p]
    #         βW_new[:,np] .+= 1
    #     end
    # end
    θW_new = 1 ./ βW_new
    return SuperMatrix(N=N, K=K, matrix=θW_new)
end

function update_αR(αR::SuperArray, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1}, NK::Int)
    N = αR.N
    K = αR.K
    # parent_nodes = [spike_list[p][2] for p in findall(x -> x>0, parents)]
    # h = fit(Histogram, parent_nodes, 1:NK+1) #count the number of times a given node is a parent
    # nodes = [s[2] for s in spike_list]
    # h = fit(Histogram, nodes, 1:NK+1)
    # αR_new = αR.array + h.weights

    αR_new = copy(αR.array)
    # for (_,n) in spike_list
    #     αR_new[n] += 1
    # end
    for (sc,(tc,nc)) in enumerate(spike_list)
        p = parents[sc]
        if p > 0
            tp,np = spike_list[p]
            αR_new[np] += 1
        end
    end
    return SuperArray(N=N, K=K, array=αR_new)
end

function update_θR(θR::SuperArray, spike_list::Array{Tuple{Float64, Int64},1}, parents::Array{Int64,1})
    N = θR.N
    K = θR.K
    βR_new = 1 ./ copy(θR.array)
    for (sc,(tc,nc)) in enumerate(spike_list)
        p = parents[sc]
        if p > 0
            tp,np = spike_list[p]
            βR_new[np] += tc-tp
        end
    end
    θR_new = 1 ./ βR_new
    return SuperArray(N=N, K=K, array=θR_new)
end


function update_posteriors!(P::SuperHawkesProcess, spikes::Spikes, P_true::SuperHawkesProcess)
    parents = spikes.parents
    spike_list = partially_observed_spikes(spikes)
    N = num_nodes(P)
    K = num_sequences(P)
    T = max_time(P)
    
    ## BIAS UPDATES
    background_spikes = spike_list[findall(p -> p==0, parents)]
    # Update α0
    α0_new = update_α0(P.bias.α0, background_spikes)
    # Update θ0
    θ0_new = update_θ0(P.bias.θ0, T)

    ## NETWORK UPDATES
    # Update αW
    αW_new = update_αW(P.network.αW, spike_list, parents)
    # Update θW
    θW_new = update_θW(P.network.θW, spike_list, parents)

    ## KERNEL UPDATES
    # Update αR
    αR_new = update_αR(P.kernel.αR, spike_list, parents, N*K)
    # Update θR
    θR_new = update_θR(P.kernel.θR, spike_list, parents)

    ## PROCESS UPDATE 
    P.bias.λ0 = sample_λ0(N, K, α0_new, θ0_new)#P_true.bias.λ0
    P.network.W = sample_W(N, K, αW_new, θW_new)#P_true.network.W
    P.kernel.rate = sample_rate(N, K, αR_new, θR_new)#P_true.kernel.rate#
end