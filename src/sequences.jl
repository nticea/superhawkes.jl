function pairwise_potential(P::SuperHawkesProcess, spikes::Spikes, s::Int, p::Int)
    spikes_list = fully_observed_spikes(spikes)
    W = P.network.W.tensor
    kernel = P.kernel

    if p == 0 # if leaf node
        return 0
    else # if has a parent 
        ts, ns, _ = spikes_list[s]
        tp, np, kp = spikes_list[p]
        return log.(W[ns,:,np,:] * evaluate_pdf(kernel, ts-tp, np, kp)) #should be (KxK). Parent dim is second dim
    end
end

function unary_potential(P::SuperHawkesProcess, spikes::Spikes, s::Int, p::Int)
    spikes_list = fully_observed_spikes(spikes)
    _, ns, _ = spikes_list[s]
    ϕ_1 = common_term(P, ns) #dimension (K)
    if p == 0  #if this is a leaf node  
        ϕ_1 .+= log.(P.bias.λ0.matrix[ns,:]) ##NOTE: I blew the effect WAY out of proportion to get it to work 
    end
    return ϕ_1 #should be (K)
end

function common_term(P::SuperHawkesProcess, ns)
    W = P.network.W.tensor
    W_ns = W[:,:,ns,:] #this spike is the parent
    c = sum(W_ns, dims=(1,2)) #sum over all child spikes on all sequences
    return -c[1,1,:] #just collapsing the dimensions, because c comes out as a tensor. Now has dimension (K)
end

function forward_pass_tree(P::SuperHawkesProcess, spikes::Spikes)
    parents = spikes.parents

    K = num_sequences(P)
    S = length(spikes)
    alphas = zeros((S,K))

    alphas[1,:] = unary_potential(P, spikes, 1, 0) #initialize the first element, by default a leaf node
    for s in 2:S #iterate through the spikes
        p = parents[s]
        alphas[s,:] = unary_potential(P, spikes, s, p) 
        # next, add the pairwise potential to the parent term 
        if p>0
            α_s2 = alphas[p,:] .+ pairwise_potential(P, spikes, s, p)'
            alphas[s,:] += logsumexp(α_s2, dims = 2)
        end
    end
    return alphas
end

function backward_sample_tree!(P::SuperHawkesProcess, spikes::Spikes, alphas)
    S,K = size(alphas)
    parents = spikes.parents
    
    #mini sampling function
    function sample_discrete(lp)
        ## NOTE: idk if I need to keep the regularization for numerical stability
        # if any(isinf.(lp))
        #     lp = exp.(lp)
        # end
        return StatsBase.sample(weights(lp))
    end
    
    samples = zeros(S)
    # Sample the last node first
    samples[end] = sample_discrete(alphas[end,:])
    # Sample backward given forward messages and samples thus far
    for s in S-1:-1:1  #iterate backward through the spikes
        lp = copy(alphas[s,:])
        for ch in findall((x) -> x == s, parents) #find the children of the current spike
            lp += pairwise_potential(P, spikes, ch, s)[:,Int(samples[ch])]
        end

        # if anneal | parents[s]>0
        #     lp = exp.(lp)
        # end

        samples[s] = sample_discrete(exp.(lp))

    end
    spikes.sequenceIDs = convert(Array{Int,1},samples) #this also updates the supernodes 
end