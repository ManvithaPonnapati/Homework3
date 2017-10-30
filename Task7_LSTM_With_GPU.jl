
using ProfileView
using Knet,AutoGrad
using Knet: sigm_dot, tanh_dot
Profile.init(delay=0.01)

type_of_array = eval(parse("KnetArray{Float32}"))  #TO ENABLE GPU
# type_of_array = eval(parse("Array{Float32}")) #FOR CPU

datafiles    = ["input.txt"]  # If provided, use first file for training, second for dev, others for test.
togenerate   = 500            # If non-zero generate given number of characters.
epochs       = 10             # Number of epochs for training.
hidden       = [128]          # Sizes of one or more LSTM layers.
embed        = 168            # Size of the embedding vector.
batchsize    = 128            # Number of sequences to train on in parallel
seqlength    = 20             # Maximum number of steps to unroll the network for bptt. Initial epochs will use the epoch number as bptt length for faster convergence.
seed         = -1             # Random number seed. -1 or 0 is no fixed seed
lr           = 1e-1           # Initial learning rate
gclip        = 3.0            # Value to clip the gradient norm at.
dpout        = 0.0            # Dropout probability.

seed > 0 && srand(seed)

# read text and report lengths
text = map(readstring, datafiles)
!isempty(text) && info("Chars read: $(map((f,c)->(basename(f),length(c)),datafiles,text))")

function createVocabulary(text)
    vocab = Dict{Char,Int}()
    # MY CODE STARTS HERE 
    
    for (char_i,unique_character) in enumerate(unique(text[1]))
        vocab[Char(unique_character)] = char_i
    end
    # MY CODE ENDS HERE
    return vocab
end



vocab = createVocabulary(text)
info("$(length(vocab)) unique chars.") # The output should be 75 unique chars for input.txt

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm_dot(gates[:,1:hsize])
    ingate  = sigm_dot(gates[:,1+hsize:2hsize])
    outgate = sigm_dot(gates[:,1+2hsize:3hsize])
    change  = tanh_dot(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh_dot(cell)
    return (hidden,cell)
end

function initweights(hidden, vocab, embed)
    init(d...) = type_of_array(xavier(d...))
    bias(d...) = type_of_array(zeros(d...))
    model = Vector{Any}(2*length(hidden)+3)
    X = embed
    for k = 1:length(hidden)
        # MY CODE STARTS HERE
        #Concatted input and hidden layer weights
        num_nodes_hidden = hidden[k]
        model[2k-1]=init(X+num_nodes_hidden,4*num_nodes_hidden) #Because we have to initialize 4w's - wf,wi,wo,wc
        model[2k]=bias(1,4*num_nodes_hidden)
        X = num_nodes_hidden
        # MY CODE ENDS HERE
    end
    model[end-2] = init(vocab,embed)
    model[end-1] = init(hidden[end],vocab)
    model[end] = bias(1,vocab)
    return model
end

let blank = nothing; global initstate
    function initstate(model, batch)
        nlayers = div(length(model)-3,2)
        state = Vector{Any}(2*nlayers)
        for k = 1:nlayers
            bias = model[2k]
            hidden = div(length(bias),4)
            if typeof(blank)!=typeof(bias) || size(blank)!=(batch,hidden)
                blank = fill!(similar(bias, batch, hidden),0)
            end
            state[2k-1] = state[2k] = blank
        end
        return state
    end
end

function predict(model, state, input; pdrop=0)
    nlayers = div(length(model)-3,2)
    newstate = similar(state)
    for k = 1:nlayers
        # MY CODE STARTS HERE
        #newstate[2k-1] is the hidden layer (look at the initweights for more explanation)
        input = dropout(input, pdrop)
        (newstate[2k-1],newstate[2k])=lstm(model[2k-1],model[2k],state[2k-1],state[2k],input)
        input = newstate[2k-1]
        # MY CODE ENDS HERE
    end
    return input,newstate
end

function generate(model, tok2int, nchar)
    int2tok = Vector{Char}(length(tok2int))
    for (k,v) in tok2int; int2tok[v] = k; end
    input = tok2int[' ']
    state = initstate(model, 1)
    for t in 1:nchar
        embed = model[end-2][[input],:]
        ypred,state = predict(model,state,embed)
        ypred = ypred * model[end-1] .+ model[end]
        input = sample(exp.(logp(ypred)))
        print(int2tok[input])
    end
    println()
end

function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end


model = initweights(hidden, length(vocab), embed)
state = initstate(model,1)

println("########## RANDOM MODEL OUTPUT ############")
generate(model, vocab, togenerate) ## change togenerate if you want longer sample text

function minibatch(chars, tok2int, batch_size)
    chars = collect(chars)
    nbatch = div(length(chars), batch_size)
    data = [zeros(Int,batch_size) for i=1:nbatch ]
    for n = 1:nbatch
        for b = 1:batch_size
            char = chars[(b-1)*nbatch + n]
            data[n][b] = tok2int[char]
        end
    end
    return data
end

function loss(model, state, sequence, range=1:length(sequence)-1; newstate=nothing, pdrop=0)
    preds = []
    for t in range
        input = model[end-2][sequence[t],:]
        pred,state = predict(model,state,input; pdrop=pdrop)
        push!(preds,pred)
    end
    if newstate != nothing
        copy!(newstate, map(AutoGrad.getval,state))
    end
    pred0 = vcat(preds...)
    pred1 = dropout(pred0,pdrop)
    pred2 = pred1 * model[end-1]
    pred3 = pred2 .+ model[end]
    logp1 = logp(pred3,2)
    nrows,ncols = size(pred3)
    golds = vcat(sequence[range[1]+1:range[end]+1]...)
    index = similar(golds)
    @inbounds for i=1:length(golds)
        index[i] = i + (golds[i]-1)*nrows
    end
    logp2 = logp1[index]
    logp3 = sum(logp2)
    return -logp3 / length(golds)
end

# Knet magic
lossgradient = grad(loss)

function avgloss(model, sequence, S)
    T = length(sequence)
    B = length(sequence[1])
    state = initstate(model, B)
    total = count = 0
    for i in 1:S:T-1
        j = min(i+S-1,T-1)
        n = j-i+1
        total += n * loss(model, state, sequence, i:j; newstate=state)
        count += n
    end
    return total / count
end

function train(model, sequence, optim, S; pdrop=0)
    T = length(sequence)
    B = length(sequence[1])
    state = initstate(model, B)
    for i in 1:S:T-1
        # MY CODE STARTS HERE
        end_seq = 1+S-1
        if end_seq > T-1
            end_seq = T-1
        end
        gradient_loss = lossgradient(model,state,sequence,1:end_seq,newstate=state,pdrop=pdrop)
        update!(model,gradient_loss,optim)
        # MY CODE ENDS HERE
    end
end

data =  map(t->minibatch(t, vocab, batchsize), text)
# Print the loss of randomly initialized model.
losses = map(d->avgloss(model,d,100), data)
println((:epoch,0,:loss,losses...))

optim = map(x->Adam(lr=lr, gclip=gclip), model)
# MAIN LOOP

for epoch=1:epochs
    @time train(model, data[1], optim, min(epoch,seqlength); pdrop=dpout)
    # Calculate and print the losses after each epoch
    losses = map(d->avgloss(model,d,100),data)
    println((:epoch,epoch,:loss,losses...))
end


println("########## FINAL  MODEL OUTPUT ############")
state = initstate(model,1)
generate(model, vocab, togenerate)

open("gpu_profile.bin", "w") do f serialize(f, Profile.retrieve()) end
