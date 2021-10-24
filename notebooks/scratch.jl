### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ bfd09b78-220f-11ec-3dc7-5386dee8523b
begin
	import Pkg
	using Distributions
	#sa = ingredients("../src/SuperNodes.jl")
	#using SuperNodes
	include("../src/SuperNodes.jl")
	using .SuperNodes
end

# ╔═╡ 26d64bc4-92b3-4d13-a7c7-c21d19a85b4c
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	return m
end;

# ╔═╡ d5713a91-532f-4431-84fd-6fff04ac215f
begin
	function to_supernode(n::Int, k::Int, N::Int, K::Int)
	    return n+(k-1)*N
	end
	
	function to_supernode(n1::Int, k1::Int, n2::Int, k2::Int, N::Int, K::Int)
	    return (n1+(k1-1)*N, n2+(k2-1)*N)
	end
	
	function from_supernode(nk::Int, N::Int, K::Int)
	    k = div(nk,N)+1
	    n = rem(nk,N)
	    return (n,k)
	end
	
	function from_supernode(n1k1::Int, n2k2::Int, N::Int, K::Int)
	    n1 = rem(n1k1,N)
	    k1 = div(n1k1,N)+1
	    n2 = rem(n2k2,N)
	    k2 = div(n2k2,N)+1
	    return (n1,k1,n2,k2)
	end
end

# ╔═╡ 27ca717c-d4be-4d5f-bf17-664872331484
begin
	N = 25
	K = 3
	data_1D = rand(Uniform(-1,2), N*K)
	n = 1; k = 3
	nk = to_supernode(n,k,N,K)
	sa_1D = SuperNodes.SuperArray(N=N,K=K,array=data_1D)
end;

# ╔═╡ 5da6cd64-70b2-4148-be98-dd8f4e26fd0c
#Test 1: Make sure that sa.array[nk] and sa.matrix[n,k] are equivalent 
sa_1D[n,k]

# ╔═╡ fb43360c-d4ca-4692-8258-f101c40f61bd
sa_1D[nk]

# ╔═╡ 13cdc45d-3fdf-4c17-9b78-534e1e9a6bc3
sa_1D.array[nk]

# ╔═╡ 4ce3dcd2-dca8-41f5-a810-32d9cdc81305
sa_1D.matrix[n,k]

# ╔═╡ 92e74355-9fde-42ea-ad5b-ebb3799e605c
#Test 2: Modify sa[n,k]. Does the change propagate to sa[nk]?
sa_1D[n,k] = 1

# ╔═╡ 96d925ca-0315-49d6-b0ed-6437db6e378e
sa_1D[nk]

# ╔═╡ 8d78e208-8424-4e32-a312-838d2f427485
#Test 3: modify sa[nk]. Does this change propagate to sa[n,k]?
sa_1D[nk] = 5

# ╔═╡ 431cc3f6-8e17-4f56-8561-7c60fe91eaac
sa_1D[n,k]

# ╔═╡ 5a79230d-be8e-4079-a6cc-9332f3ae7958
sa_1D.array[nk]

# ╔═╡ 6c2d2ae9-f2e1-4826-befe-4ee07e6861e4
sa_1D.matrix[n,k]

# ╔═╡ 26b80e69-4dbe-417a-9071-b97824bb6af1
#Test 4: Create a new array and modify sa. 
sa_1D.array = rand(Uniform(-1,2), N*K)

# ╔═╡ 433d8183-ba5b-4739-8743-1f1fddb50882
sa_1D[n,k]

# ╔═╡ 060215b8-f4d7-45df-a3c3-aa5442888b55
sa_1D[nk]

# ╔═╡ a9170046-9123-464a-8a2f-13e3a281af3d
begin
	#passed all the sanity checks! Now onto the supermatrix/supertensor
	data_2D = rand(Uniform(-1,2), (N*K,N*K))
	n1 = 1; k1 = 3; n2 = 2; k2 = 2
	n1k1,n2k2 = to_supernode(n1,k1,n2,k2,N,K)
	sa_2D = SuperNodes.SuperMatrix(N=N,K=K,matrix=data_2D)
end;

# ╔═╡ f04c87de-1a67-40bb-b1c2-f594f9e9f568
#Test 1: Make sure that sa[n1k1,n2k2] and sa[n1,k1,n2,k2] are equivalent 
sa_2D[n1k1,n2k2]

# ╔═╡ f92bd759-1dc1-4be4-b639-16951ba06df0
sa_2D[n1,k1,n2,k2]

# ╔═╡ 85c593af-2d79-4b04-a1fc-941dfc22c6b8
#Test 2: Modify sa[n1k1,n2k2]. Does this change propagate to sa[n1,k1,n2,k2]?
sa_2D[n1k1,n2k2] = 1

# ╔═╡ 350cce99-502d-470e-91aa-1e1892668484
sa_2D[n1,k1,n2,k2]

# ╔═╡ c86899ba-c3ea-4345-a14e-60453f233eff
#Do the opposite way
sa_2D[n1,k1,n2,k2] = 5

# ╔═╡ ec3d0896-34db-4e8b-b6a3-f8640130d688
sa_2D.matrix[n1k1,n2k2]

# ╔═╡ 4713dab9-e366-4501-8616-0ef68361fb34
#Test 4: Create a new matrix and modify sa
sa_2D.matrix = rand(Uniform(-1,2), (N*K,N*K));

# ╔═╡ 027edb28-fff8-42e3-84f9-aee5b03a2933
sa_2D[n1,k1,n2,k2]

# ╔═╡ 6b0804c2-ba5d-48a8-8c7b-7695764f9a56
sa_2D[n1k1,n2k2]

# ╔═╡ a36d599f-902b-41d0-abfa-2471935cc730
#It all works!! Yipee!
prior_θ0 = repeat(rand(3),inner=N)

# ╔═╡ 2db83aae-d72c-42fc-a5fd-0f833272f522
θ0 = SuperNodes.SuperArray(N=N,K=K,array=prior_θ0)

# ╔═╡ f8f43b61-8f33-4ef1-8a32-f4ad07165534
dir = Dirichlet(θ0.array)

# ╔═╡ e7dd8171-c0c6-4206-82fb-c4cec1ef8f9d
size(dir.alpha)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[compat]
Distributions = "~0.25.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e8a30e8019a512e4b6c56ccebc065026624660e8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "29890dfbc427afa59598b8cfcc10034719bd7744"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.6"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═26d64bc4-92b3-4d13-a7c7-c21d19a85b4c
# ╠═d5713a91-532f-4431-84fd-6fff04ac215f
# ╠═bfd09b78-220f-11ec-3dc7-5386dee8523b
# ╠═27ca717c-d4be-4d5f-bf17-664872331484
# ╠═5da6cd64-70b2-4148-be98-dd8f4e26fd0c
# ╠═fb43360c-d4ca-4692-8258-f101c40f61bd
# ╠═13cdc45d-3fdf-4c17-9b78-534e1e9a6bc3
# ╠═4ce3dcd2-dca8-41f5-a810-32d9cdc81305
# ╠═92e74355-9fde-42ea-ad5b-ebb3799e605c
# ╠═96d925ca-0315-49d6-b0ed-6437db6e378e
# ╠═8d78e208-8424-4e32-a312-838d2f427485
# ╠═431cc3f6-8e17-4f56-8561-7c60fe91eaac
# ╠═5a79230d-be8e-4079-a6cc-9332f3ae7958
# ╠═6c2d2ae9-f2e1-4826-befe-4ee07e6861e4
# ╠═26b80e69-4dbe-417a-9071-b97824bb6af1
# ╠═433d8183-ba5b-4739-8743-1f1fddb50882
# ╠═060215b8-f4d7-45df-a3c3-aa5442888b55
# ╠═a9170046-9123-464a-8a2f-13e3a281af3d
# ╠═f04c87de-1a67-40bb-b1c2-f594f9e9f568
# ╠═f92bd759-1dc1-4be4-b639-16951ba06df0
# ╠═85c593af-2d79-4b04-a1fc-941dfc22c6b8
# ╠═350cce99-502d-470e-91aa-1e1892668484
# ╠═c86899ba-c3ea-4345-a14e-60453f233eff
# ╠═ec3d0896-34db-4e8b-b6a3-f8640130d688
# ╠═4713dab9-e366-4501-8616-0ef68361fb34
# ╠═027edb28-fff8-42e3-84f9-aee5b03a2933
# ╠═6b0804c2-ba5d-48a8-8c7b-7695764f9a56
# ╠═a36d599f-902b-41d0-abfa-2471935cc730
# ╠═2db83aae-d72c-42fc-a5fd-0f833272f522
# ╠═f8f43b61-8f33-4ef1-8a32-f4ad07165534
# ╠═e7dd8171-c0c6-4206-82fb-c4cec1ef8f9d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
