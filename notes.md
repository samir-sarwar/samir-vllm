What am i doing:
Building an LLM inference engine with C++ and CUDA similar to a much smaller version of VLLM. VLLM is an open source LLM inference engine.Which supports many models.

Wtf is inference?:
Well from my searching the most simple answer to this is that AI inference is when a model provides an answer based on data.

Where a model is practically a file with weights that are represented as floating point numbers, with operations/functions that are using these weights. Operations are functions that take data as input and output some other data, but more specifically it's the mathematical steps a model performs like matrix math.
-> It’s obviously deeper than this with embeddings and predicting highest scoring token but from my understanding this can be abstracted when looking at inference

An LLM is a Model. One with billions of weights that let it predict the next word or token in text.

Weights are learned/discovered during the training phase. Once it becomes fixed, and we get into post training we can use them during inference to answer new inputs.

So it seems what we’re getting at here is that inference handles how we USE the model, so during training the model learns itself and discovers its weights and now that we have that, inference uses that model on a specific input to produce a prediction

Where an inference engine is the software that performs the process above efficiently. So putting the input through the math, using GPU, managing memory, and getting result

Chef Anlogy:
A good way to think about it thats helping me wrap my head around this is a chef whose learned millions of recipes.

Training: You can say culinary school, this is where they learned the recipes by seeing whats right, wrong, and correcting itself. By the end they know what certain amount, and type of ingredients will create what dish, These would be the weights. The Model is thus the chef’s learned skill and knowledge

Inference: Is asking the chef to make a meal with some ingredients and the chef producing an answer, so see this is quite abstracted. Give him some shi, and see what he cooks up (literally). Inference is the actual act of cooking.

Inference Engine: The kitchen; providing pretty much everything needed for the chef to execute the meal based on the input, and based on what they’ve learned from school. The better this is, the more efficient. I.e it affects speed, memory use, cost, and how many requests it can handle. This is easy analogous to kitchen being the inference engine as well. But an important note would be that it doesnt affect the quality of the output, this is dependant on the model and what is learned during training phase.

Thus you may ask, why do we need an inference engine beyond optimizations? Well you can’t run the file with the weights of the model, this is not executable, its just a bunch of numbers. The architecture of the model also can’t run its just a blueprint and not that useful without the model being trained by data. And so we must Serve the model meaning writing a program that implements the operations of the model.

Why C++ and CUDA:
Well, we wanna be fast, and optimize hardware as best as possible, more specifically the GPU. C++ is extremely performant with insane customizabiltiy with optimizations, and CUDA allows us to program GPU hardware and memory. Why GPU? Remember above when I mentioned that the cook needs to actually create the meal during inference, and so it needs to perform all of the model operations which is a shit ton of matrix multiplications that are handled much better on GPUs rather than CPUs as they have more powerful processing cores.

SafeTensors:
There are different formats in which you can download models, SafeTensors being of the most popular formats.

A safetensor file is made of 3 sections, header size, JSON header, tensor data. A quick aside; a tensor is a multi-dimensional array or numbers, where tensordata here represents the model weights.

Header size is always 8 bytes, each unsigned 64 bit, The header is in JSON meaning its just a group of key value pairs. Where the key is a unique tensor name. Every value is a JSON itself containing 3 keys; dtype, shape, offsets. Dtype says what data type tensor is stored in. shape 64 says the dimensions of the tensor, as i mentioned above a tensor is a multidimensional array. Offsets say where the tensor is stored within the tensors data section. It is a start and end to show where the tensors raw numbers are stored in the files data section.

What we implement:
LlamaForCausalLM(
(model): LlamaModel(
(embed_tokens): Embedding(128256, 2048)
(layers): ModuleList(
(0-15): 16 x LlamaDecoderLayer(
(self_attn): LlamaAttention(
(q_proj): Linear(in_features=2048, out_features=2048, bias=False)
(k_proj): Linear(in_features=2048, out_features=512, bias=False)
(v_proj): Linear(in_features=2048, out_features=512, bias=False)
(o_proj): Linear(in_features=2048, out_features=2048, bias=False)
)
(mlp): LlamaMLP(
(gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
(up_proj): Linear(in_features=2048, out_features=8192, bias=False)
(down_proj): Linear(in_features=8192, out_features=2048, bias=False)
(act_fn): SiLUActivation()
)
(input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
(post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
)
)
(norm): LlamaRMSNorm((2048,), eps=1e-05)
(rotary_emb): LlamaRotaryEmbedding()
)
(lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)

This is pretty much the entire model architecture for Llama 3.2 1B instruct. What does that mean? Pretty much just outlines the different operations we need to run, as well as the data shapes and types that we need to use.

From this we don’t actually know the order of operations or data type, it’s just an object with code that I don't really understand yet. We can search up the model on huggingface and we can find out that the dtype for the model weights are BF16
-> BF16 is a 16 bit floating point number, we use it because its a compact format that allows us to still access the full range of a 32 bit float, it is separated into sign bit, exponent bit, and fraction bits, both a 32 bit float and 16 bit float have the same number of exponent bits meaning they cover the same range of floats (-(2^8), 2^8 -1) i think thats right I could be wrong on that. Where they differ is the amount of fraction bits, meaning pretty much that the tradeoff of using this more compact format is a bit of precision. For inference this is a positive tradeoff, this small loss of precision is worth the benefit from effectively halving memory use.

Like I think I mentioned before, we need to know the order of operations to effectively produce correct output from the model, as the operations are different computations in certain order to produce a certain output.

For this we need to search up the LLM architecture, diagrams are good and Sebastian Raschka has a bunch of them here or just search it up.
Lets figure out how I’m supposed to dissect this diagram though.

Figure 1: Llama 3.2 1B Architecture Diagram [Sebastian Raschka]

So we can see that we start off by sending some input text to the model. We then turn them into tokens, we do this using a Tokenizer , this converts text into pieces the model recognizes. So in this phase the tokenizer takes our input and splits the sentence or phrase into smaller pieces, which could be whole words, pieces of words, punctuation, etc. Then it assigns IDs to these specific pieces, it does this because it created a fixed list of token pieces during its training, each mapping to an ID, the ID value itself is arbitrary. You might think, okay well what if it wasnt trained on a certain input, well like i mentioned before then the tokenizer would break it down as much as possible (split ex.: [“hyperglob”] -> [“hyper”, “gl”, “ob”]) until it resembles something part of the tokenizers vocabulary. Then we hit up a mega table with all of the model weights, called the embedding table, where the Token ID we got from the previous step represents a row in the table. (embed_tokens): Embedding(128256, 2048)
This is the code that tells us that there are 128,256 possible token IDs and 2048 learned weights, which form an Embedding Vector . That pink box, is called a transformer block also called layers, there are 16 of them that have a shit ton of operations that I don’t know yet but will get a clear understanding as I build more of the engine. They consist of these computations:
RMS Norm
Residual connection
Masked grouped-query attention, which consists of:
Q projection
K projection
V projection
RoPE with Q projection
RoPE with K projection
Attention
Attention scores
Causal mask
Softmax
Residual connection
Attention scores with V projection
O projection (output projection)
Residual connection add
RMS Norm
Feed forward (like in first neural networks, Multilayer perceptron), which consists of:
Gate projection, first linear layer
Up projection, second linear layer
SiLU activation function, similar to ReLU but it's looking more like a sigmoid
Down projection, third linear layer
Residual connection add
I truly dont got that much of a clue of these computations in the layer but will understand later. Then a final RMS Norm, a Linear output, then Argmax. Ngl still just copying from the diagram and docs. Will learn as I go.

An aside on memory:
Data can live in host or device. Host would be the actual PC using the CPU, it has slow big memory DRAM, DRAM is separate from the CPU but the CPU has its own very quick but small on chip memory called SRAM. This is where we have stuff like L1, L2 cache which allow for quicker access to recently accessed memory. Device would be the GPU, it also has slow big memory called VRAM, and also has quick small memory SRAM. Like I mentioned before we want to be doing the large computations on the GPU rather than the CPU, but that means we need to access all of the information like model weights, input tokens, and intermediate results, but the GPU can’t directly access the Host DRAM. So what we do is copy the data from CPU DRAM -> GPU VRAM. This isn’t as easy as it sounds at the low level though. On the cpu we have to create a variable and write to it, the compute the size of memory taken by the variable, i.e how many elements are stored, then multiply that by the size of the variable type and thus you know the total amount of bytes. Then on the gpu allocate memory so we have somewhere to put our stuff, then copy it over, now we can effectively use this data in GPU computations as its on our device VRAM. For the GPU memory allocation and copy there are CUDA specific functions which are quite similar to ones I worked with in C, like cudaMalloc, and cudaMemcpy.

Loading the Model:
It’s time to start work and the first step when working with inference is to actually load the model you’re working with. Remember how we talked about above how we have to copy the weights from CPU to GPU memory so we can use the GPU larger VRAM, thats effectively what we’re doing now. Remember how before we said that a model is actually a single file with weights, thats pretty much what we’re reading now, a model.safetensors file. Remember before that A safetensor file is made of 3 sections, header size, JSON header, tensor data. Where tensor data is actually many tensors. So we have to read from safetensors file, to do this remember in C how we had to create actually open a file using fopen or open(), its the same thing in C++ but now we use std::ifstream, meaning input file stream which creates an object that lets cpp program read data from the file. We also have to use ios::binary to say the file is raw binary data so it preserves every byte exactly as if we dont sometimes the compiler tries to optimize or translate special chars or line endings while reading which would corrupt bytes and make offsets wrong.

It’s been a minute since I updated my notes but I was pretty much just learning as I coded through loading the model. So it seems we left off on learning to open files using ifstream. Once we open the model file, we do in fact need to read it as well. As we’ve already mentioned numerous times, the file is a safetensor file. Which is formatted by the header, the JSON header which includes pretty much meta data on every single tensor in the model. There are 16 transformer layers, each of which have 9 different tensors inside of them. These 9 different tensors occupy some space in that file, these are recorded by their offsets, which can be found in the JSON header. This is important because at some point we’re going to have to use these model weight values and we can’t keep reading from the file. Super slow! We have to bring it to the GPU memory (VRAM) So how? Now we’ve opened the file which is the first step, but we also have to read it. Why? Because that can tell us a bit more about how we should be specifically parsing this. As I mentioned before each of the tensors are multidimensional arrays of floating point values that are weights for us. Each having a unique name as their key that defines what the tensor is actually for. Meaning if later on we need to find these tensors knowing the naming convention allows us to be able to quickly find what we’re looking for. We’ll get into that a bit more later, but thats just a preface on the motivation behind reading and moreso printing what we find from reading the JSON header that describes each tensor. So how do we read? Well first we have to read the original header, a safetensor file starts with the header size as a uint64_t meaning 8 byte value to tell us how large what we’re reading next is going to be. Then we call .read(destination, bytes to read) on the safetensor file that we opened. It will return to us the value of the first 8 bytes as we had set it that way. Now we know how big the json header is so we can read it to completion. Now that we know we can use the .read again but now for the next headersize amount of bytes. One thing to mention, we are calling read again on our safetensor file, meaning the file pointer is currently at the start of the json header anyways, i.e 8 bytes in so we don’t have to handle for that. But what do we read to, read needs a destination buffer, of type char, an array of chars, which is a string. So we make a string of size headersize, initializing it to \0. Then we read headersize amount of bytes to it. Great! Now we have the JSON header, and we can parse it for the offsets that we will need for 2 things. 1) The max offset will tell us how much space the tensordata takes up, as remember we know how big the header is but not the raw tensor data i.e the weights. 2) It would allow us to read the starting offsets, which we need later to create pointers to them, as we will be allocating all this memory to the gpu, and we want pointers on our cpu memory so we can quickly access the gpu memory. Okay, so how do we go about parsing?
The way we go about parsing is: we’ve already created a string which has our JSON header in it. We use the JSON library that we imported before to convert our string, header, into a JSON object. We do that using the json::parse function.
As I mentioned before, we need to remember the starting offset so we can create our pointers later. The way we can go about that is using a hashmap, which in C++ is an unordered_map. The key-value type would be a string for the key, and a uint64_t for the value.
We also initialize our max_offset at zero because we will continuously check that throughout each iteration of a for loop that goes through the items of the header, which is now a JSON object.
We do this using a structured-binding for loop, which will go through the name and tensor_info for each JSON-header item. This is where having printed the header before would be useful, as you would know exactly what we are looking for when we need to find specific values from the JSON object.
For example, the data offsets are found at "data_offsets" in the JSON object. data_offsets itself is an array of two values: the start offset and the end offset. Those can be found by using data_offsets.at(...), then converting them into a uint64_t, which is a 64-bit unsigned integer.
Then we use emplace. That essentially adds those values to our hashmap, and then we recompute max_offset by checking the current tensor’s end offset against the current maximum offset.
Now we have a hashmap of our offsets which is good, but we still need to actually copy the full tensor data to memory so we can copy it to the GPU.
That part is actually relatively simple due to what we did with maxoffsets. Since the file ptr is already at the start of the raw tensor data and we know how large it is, we can just simply call read again.
Brings about the question though, where should we store this. We will use a vector of type char, as that is what .read expects but also it matches the file’s exact raw byte layout. If you remembered from before yes we said the actual floating point values are BF16 which is 16 bits, meaning simply they are just two contiguous 8 bit chars.
Now all the raw tensor data is in memory, and we must copy it to the GPU but before that we need to allocate space on the GPU. Use cudaMalloc which will allocate a space on the gpu and set a pointer to point to the start of the GPU block reserved for this data. Hence before we allocate we need to initialize a void pointer as null. So it can be assigned a value. Then we use cudaMemcpy to actually run the memory copy. Has pretty simple params that we’ve already made so it works out nicely.
So initially, the way I went about is creating a hashmap of pointers each pointing to where in GPU memory that weight was. However, this would need string construction, hashing, and a hashmap lookup. Which is actually fine on initial run. As this is just first phase on loading, however when actually running inference we will have to go through all 16 layers of this model meaning if we were to go with this approach it would go through alot of string constructing and hashing to get the actual pointer which is slow. Instead by creating a struct,that has all the potential values for each layer and then indexing by that layer we get a much quicker lookup as well as no string construction when going through each layer.
Though that means we need to handle that construction on initial load instead. So think about it as we create BF16 type pointers to specific values, then the ones that repeat in every single transformer layer we create an array of 16 where each value is a BF16 pointer. To that layers specific tensor.
Then we loop through each layer and do the math, see where each pointer should point to and save it for that value in the struct so it can be called from cpu memory later on.
First though, we get the base by seeing where the memory points to in the gpu as we got that returned to us before from cudaMalloc and there is a pointer we can store it in, in the weights struct. Then we calculate the location by using the base and the offset hashmap from before and do this for each tensor in each layer.
By the way, we need to use weights outside of this function, so the way we did that was passing it by reference into the loading function itself.
The struct approach works because we are building this specifically for Llama 3.2 1B, and we know the model architecture, and we printed the JSON header to see how tensors were named.
Tokenization
We covered this briefly before but not to the max extent, it is pretty much the process of taking the input text, and splitting it into many parts, each of these parts receiving a token ID through tokenization. These token ids are what is used to look up the specific token in our embedding table, that row of values (vector) is the tokens starting representation. The transformer layers then use attention and other operations to update hidden state vectors based on the surrounding tokens, producing contextual meaning. Ok so when it comes to actually using a tokenizer often when a model is released it will come with a tokenizer already and or there are open source libraries that will provide them. So it starts off with getting an actual tokenizer file, this is given with the model itself. Think about it like this, you know how our LLama model was trained and its output was a simple safetensors file with a bunch of weights, thats the output of the tokenizers training with tokenizer model file. The tokenizer was trained with byte pair encoding which pretty much means, pieces of text were split into UTF-8 bytes, each byte as a separate piece and then you repeatedly merge adjacent bytes that show up frequently in training data, then give each merged piece a token ID. Anyways, now we have all these pieces of text and their token ID values, and this is what gets stored in the tokenizer file. The tokenizer vocabulary is basically the tokenizer’s dictionary: it contains every text/byte piece the tokenizer knows about, along with the fixed token ID assigned to that piece.

So when we actually give the model a prompt like “Hello world”, the tokenizer library first looks at the text and splits it up based on its rules, like spaces, punctuation, words, numbers and things like that. It then turns those pieces into UTF-8 bytes. UTF-8 is just the standard way computers represent text as numbers from 0 to 255 called bytes, so it can handle normal English text, punctuation, emojis and other languages all using the same system.
It then starts checking which byte combinations exist in the tokenizer vocabulary. It will keep merging byte pieces together based on the BPE rules it learned during tokenizer training. So instead of every character always being its own token, common things like “ hello”, “ing”, “tion”, or even whole words might already exist as one token. Once it has found the final pieces, it looks up the token IDs that were assigned to them when the tokenizer was trained.
So the final result is just a list of integers, something like [128000, 9906, 1917]. The tokenizer does not actually do any neural network calculations itself; it is really just translating text into the integer IDs that the trained Llama model understands.
Then our C++ inference engine takes those IDs and copies them to the GPU. From there, the embedding lookup uses each ID to grab the correct row from the embedding table in our safetensors weights. Those embedding vectors become the starting hidden states that go through all of the Transformer layers.
At the end, the model gives us a new token ID that it thinks should come next. We then use the tokenizer in reverse, where it looks up that ID in the vocabulary and turns its stored bytes back into readable text.

input text
→ tokenize → token IDs → embedding lookup → Transformer layers
→ next token ID → tokenizer decode → output text

Normilization using RMSNorm:
As we saw from the diagram above (Figure 1) the next phase of our inference engine is for our tokens to go through RMSNorm1 which seems to mean the first Root Mean Square normilization, furthermore it seems to be happening inside of a Transformer layer, meaning its a repeated calculation for each layer. What is it/ Why do we do it, so firstly the mathematical operation of RMS Normilization is pretty much just making the average of the vector closer to 1, which in essence also gets each of the values in the vector closer to one. Why do we do this, and doesn’t this affect our weights? As I mentioned earlier, how we look at the vectors from the embedding table is that their semantic value is embedded in its direction. A practical example would be if say the word woman, man, brother, and sister each had its own token ID in our vocabulary. If you were to take brother, subtract the man vector, then add the woman vector, the vector would point in the direction of sister. All this while saying the actual magnitude didn’t matter, just the direction. So if we were to normalize this, meaning the relation between the values in the vector would be the same but their actual magnitude would be less. The actual calculation is very simple. We just take each value in the vector, square it, then get the average, then get the square root of this average. Lastly divide each value in the vector by by the RMS value.
Ex. [1,2,3,4,5] -> [1,4,9,16,25] -> μ = 11 -> = 3.316
-> [0.302, 0.603, 0.905, 1.206,1 .507].
We went over how it works, why it works, but why do we do it? Well its because due to the fact we repeatedly multiply these weights, its possible they either spiral towards infinity, or shrink towards 0. But we want it to be stable so the neural network can learn effectively. You may have heard of LayerNorm instead of RMSNorm, which makes sense it was the OG, part of the paper Attention is All you Need. LayerNorm does two things: it re-centers the data (by calculating and subtracting the mean) AND it scales it. Researchers later discovered that the re-centering step didn’t actually help the model learn better, and so RMSNorm was more popularized.

Understanding the code:
RMSNorm treats each token IDs 2048-value hidden vector (vector that comes initially from the embedding table, These are called hidden values because they are the model’s internal representation rather than directly interpretable outputs. ) independently. Before we do anything, as we did when loading the model. Computations are to be done on the GPU so we have to allocate and copy over the actual vector of token IDs to the GPU using cudaMalloc and cudaMemCpy. Then we are going to have to deal with a table of all the token IDs and their respective hidden values, as afterall these are the weights we are going to be working with. And so you allocate space on the GPU for that too, you dont re-create it on cpu to copy on to the GPU though, as if you remember correctly our preliminary work to put all the weights on the GPU as well as create a pointer directly to the embedding table was already done, no need to duplicate efforts. Now we’re going to be writing a CUDA kernel, which is pretty much just a way of saying a function that will run on the GPU. But instead of running just once, the GPU is going to take this exact same function and run it thousands of times simultaneously across all of its cores. Taking complete advantage of parallelism. **global** declares a CUDA kernel. It runs on the GPU and is launched by CPU code using <<<blocks, threads>>>. Passed into our kernel is a ptr to the input tensor with token_count rows, and 2048 columns. As well as an identically sized output tensor for normalized values. Lastly a weight vector containing 2048 weights, each token uses same weight vector. In this kernel, we face an interesting parallelism problem, where the threads actually have to communicate with each other. In this case its due to the fact that you require a total sum for one of the steps of the computation of RMSNormilzation vector. One thing about CUDA kernels is that you can only at once spawn 1024 threads in a block, this is important as we can only safely share partial sums amongst threads within the same block. Meaning 2 blocks won’t work so the solution would be to have each thread handle two of the token’s 2048 hidden values and stores their squared sum in its spot.
Ex: partial[tid] = x0 _ x0 + x1 _ x1;
Which brings us to **shared** which pretty much just means shared memory, that any of the threads in the block can read/write to. It is in float instead of bf16 as here with all these computations we want to assure as much precision as possible before its lost later on, if we compute using bf16 now it’d be farther away from the true value. One block handles one token. It stores each threads partial sum. The example illustrated above is not necessarily how we take the value though. We do it by taking it from the first half squaring it, and adding from the second half squaring it and putting that value in the partial sum vector. So it’d be 1024 away from base + tid. Then we can join the threads together. Think of \_\_syncthreads() like pthread_join(). Now that we did that we know partial vector is filled up completely. To find the maximum, if we went through each value, it would take long, instead we can take advantage of parallelism and use Tree reduction algorithm. You can thing of it as instead of one person adding up a list of 8 numbers one by one (which takes 7 steps), a GPU assigns multiple threads to add pairs of numbers simultaneously.
Ex:
Step 1: 4 processors add 4 pairs of numbers at the same time. (You are left with 4 results).
Step 2: 2 processors add those 4 results in pairs. (You are left with 2 results).
Step 3: 1 processor adds the final pair to get the total.
The Result: It only took 3 steps instead of 7.
In our case we would have 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1. So in the first round we’d have:
partial[0] += partial[512]
partial[1] += partial[513]
...
partial[511] += partial[1023]
Now, there are 512 useful sums, then we keep breaking it down until the total sum remains in partial[0]. We then join the threads again, but why? It works as a barrier, this is a mechanism in concurrent programming to ensure one thread does not move on to the next iteration of the loop before the first one is complete. Ensuring correct ordering of data read.
RoPE (Rotary Positional Embedding/Encoding):
The motivation behind this comes from the fact that a transformer is unable to tell the difference in the meanings of words by their position in a given statement. An example of this could be:
“I bought an Apple watch” and “Watch an Apple I bought”
In this example the input is made up of all of the same tokens except in a different order, does that matter? Yes, because in this case apple refers to two different things here, but if Apple itself has one embedding vector it is true that the vector is the same regardless of where in the input prompt it lives. Lets do a short run-through of what exactly happens during the transformer layer. Each token is multiplied by Query, Key, and Value matricies to output their respective Q,K,V vectors for that embedding token.

Looking at apple as being x4, with q,k,v vectors q4,k4,v4 respectivley. You get the attention vector for apple by taking the dot product of q4 and kT then divide by sqrt(dk) and apply softmax, just as we saw in the attention formula, just this time for one vector instead. Lets say, we did not use masking here, and we were computing for an output vector after multiplying by value vectors, if we were to shuffle the words around as we did earlier with the example, then the weighted sum would be the exact same, meaning the transformer would see both sentences as the exact same despite them meaning totally different things. Original transformer paper introduced a sinusoidal positional encoding method where each position is assigned a unique vector of periodic components with varying frequencies. We then need to inject this positional information into the embedding vectors, and there are multiple ways of doing it, like concatonating them, adding them, or use MLP. Original paper uses adding.
Absolute positional encoding is when we add a unique positional vector to each tokens embedding, but this faults when we take two sentences with the same input tokens but rearranged that actually mean the same thing. In this case the transformer would believe that they mean totally different things due to the differences formulated from the positioning vectors. Hence why we use Relative Positional Encoding. We rather encode the relative position between tokens, as now we’re looking at the position relative to the other tokens solving the problem mentioned earlier.
Now lets finally get more into RoPE. For simplicity of understanding imagine a two dimensional query vector for the token dog, RoPE rotates this vector in R2 by θ depending on its position in the input. It only depends on the position, meaning we do not care how many or what tokens come after it when we are encoding.

Lets look an example of RoPE. In Transformer models, attention between words is calculated using the dot product of a Query vector (the word looking for context) and a Key vector (the word providing context). Lets say a sentence has ‘I’ and ‘dog’ where dog is the query vector and I is the key vector. In this case that means wes are looking at how much attention the token ‘dog’ pays to the token ‘I’. Let’s look at the sentence “I walk my dog”, where I is at pos 1, and dog is at pos 4. Meaning their key and query vectors are rotated by 1θ and 4θ respectivley. How are they rotated, well, they are multiplied by this matrix. ->  
Then when we go to calculate we dot product as per normal then we rotate by the angle difference so 4 - 1 =3θ. Ok well how is this relative? If we were to add more words to the start of the sentence, their individual magnitudes of rotation would be greater but when forming the dot product and taking the difference in angles it’d be the same as before thus, relative encoding as we talked about earlier.  
Now that we’ve seen an example, we’re ready for a general formulation, where

Here, m denotes the position at which the vector is in, in the input prompt. Thus I assume that m is bounded, 0 < m <= token_count, where : m E Z. Now if we were to calculate the attention between two vectors, with query at pos m, and key at pos n, and attention vector = dot product of query and key. Using the knowledge of how the key vectors are transposed in the attention formula, we’d also transpose the rotation matrix, meaning it becomes -n, thus the angle difference we explained earlier.

This simplified example only works in the 2D case though, in the case of higher dimensions we split them into groups of smaller ones with rotanial encodings then combine after. But this takes alot of memoery, so we can do element wise operations. Note: lots of examples coming ->
Here is one tiny RoPE example with one 2-number pair. Real Llama heads have 32 pairs, but the same calculation happens for each one.
Assume the query for token A is:
Q_A = [ 2 ]
[ 1 ]
Assume the key for token B is:
K_B = [ 3 ]
[ 4 ]
Let their positions be:
p_A = 2, p_B = 5
And, just for this example, let this pair’s RoPE frequency be:
ω = 0.1

In reality we use the frequency formula we see in the code for each pair,
RoPE calculates one angle per token:
θ_A = p_A × ω = 2 × 0.1 = 0.2
θ_B = p_B × ω = 5 × 0.1 = 0.5
Now rotate the query:
Q'\_A = [ 2·cos(0.2) - 1·sin(0.2) ]
[ 2·sin(0.2) + 1·cos(0.2) ]
Using:
cos(0.2) ≈ 0.9801, sin(0.2) ≈ 0.1987
we get:
Q'\_A = [ 2(0.9801) - 1(0.1987) ] = [ 1.7615 ]
[ 2(0.1987) + 1(0.9801) ] [ 1.3774 ]
Now rotate the key:
K'\_B = [ 3·cos(0.5) - 4·sin(0.5) ]
[ 3·sin(0.5) + 4·cos(0.5) ]
Using:
cos(0.5) ≈ 0.8776, sin(0.5) ≈ 0.4794
we get:
K'\_B = [ 3(0.8776) - 4(0.4794) ] = [ 0.7150 ]
[ 3(0.4794) + 4(0.8776) ] [ 4.9486 ]
Attention now compares the rotated query and key with a dot product:
Q'\_A · K'\_B = (1.7615)(0.7150) + (1.3774)(4.9486) ≈ 8.075
That 8.075 is this pair’s contribution to the attention score.
The important part is that token A was rotated by 0.2 radians and token B by 0.5 radians. Their difference is:
θ_B - θ_A = 0.5 - 0.2 = 0.3
which is also:
(p_B - p_A)ω = (5 - 2)(0.1) = 0.3
So the comparison naturally contains their relative distance:
p_B - p_A = 3
In a real head, this happens for all 32 RoPE pairs, and the attention dot product adds all of their contributions together.

So to put it simply all we’re doing is rotating each weight in our query and key vectors for each of our tokens, they rotated based on which token position they have in the prompt, and then finally as you may be thinking dont we need to get the relation between all of them, yes we do that when we finally compute the dot product and the final attention scores go into its respective matrix, happens during attention.

How it works for each of the rotations within the vector, query OR key itself is as such:

For one token, the flow is:

token embeddingQ, K, V
Take that token’s Q vector. One head has 64 numbers:
Q=[q0,q1,,q31,q32,q33,,q63]
RoPE makes 32 coordinate pairs inside that one head:
(q0,q32), (q1,q33), , (q31,q63)
Then it rotates every one of those pairs using that token’s position.
So for token 3:
(q0,q32)rotateusingposition3

(q1,q33)rotateusingposition3

(q31,q63)rotateusingposition3
That gives one finished position-aware query vector:
Q'3
The same happens to its key vector:

K3K'3

Only after every token has its own rotated Q and K does attention compare tokens:
Q'3K'1

Q'3K'2

Q'3K'3
More examples just to drive home the concept:
Tiny fake head with 4 numbers instead of Llama’s real 64:

Q3=[2, 5, 3, 7]
Using the same half-split layout as your code, its RoPE pairs are:

pair0=(2,3)

pair1=(5,7)
Now give each pair a different frequency:

0=0.1

1=0.01
Both pairs are from token position 3, so:

3,0=30.1=0.3

3,1=30.01=0.03
Pair 0 rotates more because its frequency is faster:
(2,3) → [ 2·cos(0.3) - 3·sin(0.3) ] ≈ [ 1.024 ]
[ 2·sin(0.3) + 3·cos(0.3) ] [ 3.457 ]
Pair 1 rotates only a little because its frequency is slower:
(5,7) → [ 5·cos(0.03) - 7·sin(0.03) ] ≈ [ 4.788 ]
[ 5·sin(0.03) + 7·cos(0.03) ] [ 7.147 ]
So the rotated query for token 3 becomes approximately:

Q'3=[1.024, 4.788, 3.457, 7.147]
Same token position; different pair frequencies; different rotation amounts.
In the real model, this happens for all 32 pairs in a head.

Remember though this is how the frequencies are actually calculated:

i=12i/d

=500000

d=64

Theta and d values coming from how LLama is.

Last thing to note, is that in LLama we also scale the frequencies with pairs, why and how we do this noted below:

Each RoPE pair has a frequency i, which determines how quickly it rotates as token position increases.

anglep,i=pi
A pair’s wavelength is the number of token positions needed for one full rotation:

i=2i
Therefore:

largeishortwavelengthfastrotation

smallilongwavelengthslowrotation
Llama divides RoPE pairs into three wavelength regions:

i<2048
Keep fast frequencies unchanged:

'i=i
These pairs preserve detailed positional differences between nearby tokens.

2048i8192
Smoothly blend between original and slowed frequency:

s=8192/i-14-1

'i=(1-s)i32+si
This prevents an abrupt change between neighboring pairs.

i>8192
Slow the already-slow frequency:

'i=i32
This stretches its wavelength:

'i=32i
Why: fast pairs retain precise local position information, while stretched slow pairs remain useful across much longer contexts.
Attention:
Single Headed Attention:
So this is the big kicker, and what you’ve probably heard a million times, especially on the famous paper ‘Attention is all you Need’ that really kickstarted the whole AI boom. In our engine we use Multi-headed attention but its good to start here to get a foundation. We may have jumped the gun a bit by talking about RoPE before really understanding attention, so I may change up the order of where this is in the notes.

Pretty much though, you can think of it as what powers context within the prompt, it allows us to semantically understand what a word/token is trying to say by looking at other words in the prompt. A good example could be ‘I just clean windows’ for us humans thats pretty easy to understand, but if we were to just use regular feed forward network or something, theres no way of us to know if windows, mean actual windows, or the software windows meaning if we had ‘I just downloaded windows’ it’d think of windows in the same way despite the whole meaning being drastically different. Attention allows us to use other words in the prompt to actually understand what certain words mean. It’s like if I were to ask you as a human how did you understand the difference between the two sentences, you’d say it’d be the words clean and downloaded, this is precisely what we can do with attention, but how?

The way we can measure the actual relevance from token to token would be through a vector dot product, you are already familiar with how we get token embedding vectors from the learned embedding table, and that’s now a key part of what we do here? If you’re keen you may have thought, why don’t we just do the dot product of one token, by the other tokens to measure their similarity, the problem is often words that might highly relevant to each other end up having very different token embeddings, furthermore it can be asymmetrical meaning one word can mean alot more to another word then the other way around, an easy way to think of this is an adjective and a noun: ‘Amazing food’ in this case their token embeddings would probably be very different leading to low similarity, as well as the fact that for food, the word Amazing is very relevant, but for Amazing it doesn’t really care about the word food.

Rather than compare token embeddings directly, the model first learns two different representations of every token: a query and a key. The query represents what the current token is looking for, while the key represents what information that token can offer. We compute these values by multiplying a tokens embedding vector by a Query (W_q) and Key (W_k) matrix. W_q learns which combinations of a token’s current features describe what it needs to find. W_k learns which combinations describe what that token can provide. Their dot product measures whether what one token seeks matches what another token offers.

Then, we simply take our token embedding vector and multiply it by these matrices to give us new vectors to use to calculate how much ‘attention’ one token pays to another. Multiplying it by those matrices can be called Q,K,V projection and it’s what we do in cuBLAS, in Video 6 of the series!

Perfect, now you can think of their attention scores as being the dot product of the query and key vectors. The full attention score matrix can thus be given as

S = QKT + M  
Where M, is a mask as we don’t want to consider a key vector that comes after the query itself. We then divide by a scalar sqrt(dₖ) where dₖ is the size of one key vector in a single attention head. We use it because a dot product gets larger on average when it adds more dimensions. Without scaling, the attention scores could become very large, making softmax overly confident where almost all probability goes to one token. Lastly, a problem we can face is that this dot product could result in a negative value and its hard to infer what exactly that means so you can apply an exponential function to make them non-zero, and then normalize them so their values are within a set scale. This is exactly a process of softmax, thus the Attention Weight Matrix can be given as
A = softmax(QKTdk + Mask)
The attention-weight matrix tells us where each token should look, but it doesn’t yet tell us what information to take from those tokens. That is the purpose of the value vectors. Each value vector is created from the token’s current representation using its own learned matrix:
V = X × Wvᵀ, you can think of values as the actual information that token can provide. Q and K determine the attention weights; V contains the information we transfer once we know which tokens matter.
We therefore multiply the attention-weight matrix by the value matrix:
O = A × V
For a token i, its output is a weighted sum of the value vectors of the tokens it attends to:
o_i = A[i,1] × v₁ + A[i,2] × v₂ + ... + A[i,i] × vᵢ
Or… o_i = Σⱼ A[i,j] × v_j

A high attention weight means that token’s value vector contributes more information to the new representation of token i. The output O is therefore a new context-aware representation for every token.

Multi-headed Attention:
So now that we understand single headed attention we know that one head can look at all the earlier tokens to work out which ones are relevant, and then take a weighted mixture of their value vectors. But then, why not just use one massive attention head and call it a day?
The problem is that one attention head has to use a single definition of what “relevant” means. In a sentence, a token might need to care about multiple things at once, like maybe one earlier word tells us the subject, another tells us the tense, whereas another explains what a pronoun refers to, and another provides the actual topic. Having one head do all of this would be like asking one person to look for every possible pattern at the same time.
Multi-headed attention lets the model look at the prompt from multiple different perspectives at the same time. Rather than creating one query, key, and value vector, we just split them into multiple smaller q,k,v vectors, with one set for each head. Each head has its own learned section of the Q, K, and V weight matrices, so each can learn different useful patterns.
For our Llama 3.2 1B model, the query projection has 2,048 values per token. We split this into: 32 query heads × 64 values per head = 2,048
So, after the Q projection, we can think of it as:
Q = [Q₀, Q₁, Q₂, ..., Q₃₁]
where every Qₕ is a smaller query matrix with shape: [# of tokens, 64].
The same idea then also applies to keys and values. Each head runs the exact single-head attention calculation independently:
Sₕ = (Qₕ × Kₕᵀ) / √64 + M
Aₕ = softmax(Sₕ)
head_outputₕ = Aₕ × Vₕ
Each head can now attend to different earlier tokens! One of the heads could learn that the food token should look at Amazing as a key, another could care more about the earlier verb, surrounding sentence structure, or something else. The heads aren’t manually assigned jobs, instead the model just learns what is useful during training.
Once every head has created its output, we concatenate them. This just means placing their 64-value output vectors beside each other.
head_output₀: 64 values
head_output₁: 64 values
...
head_output₃₁: 64 values
32 × 64 = 2,048 values
Meaning now for a layer, we have a 2048 valued multi head attention output per token We then multiply it by another learned weight matrix, Wₒ, called the output projection:
attention_update = Concat(head_output₀, ..., head_output₃₁) × Wₒᵀ
Wₒ lets the model mix the useful information discovered by the different heads. Finally, we add this attention update back to the token’s previous hidden state using a residual connection, you can think of the attention update as being Δx and we are adding the change to the old embedding token to get new hidden state
new_hidden_state = old_hidden_state + attention_update
Or… x’ = x + Δx
Grouped Query Attention:
One interesting part of LLama’s model architecture is that it uses Grouped Query attention, which pretty much means it splits key and value heads amongst multiple query heads. In our LLama 3.2 1B case in specific it has 32 query heads, but only 8 key heads and 8 value heads. This means every group of four query heads shares one key head and one value head:
Q heads 0–3 use K,V head 0
Q heads 4–7 use K,V head 1
...
Q heads 28–31 use K,V head 7

So now our actual calculation for each query head h would be:

kv_head = h / 4

Sₕ = (Qₕ × K_kv_headᵀ) / √64 + M
Aₕ = softmax(Sₕ)
head_outputₕ = Aₕ × V_kv_head
The reason we do this is that it keeps the multiple query perspectives that make multi-head attention so powerful, while using fewer key and value vectors. This saves a ton of KV-cache memory during inference. We’ll get into why that’s so important later on ;)

Feed Forward Network:
Ok perfect, so up until now, within the full model architecture of LLama, we’ve completed the attention part of one individual layer, you can call the first one layer 0. What does that mean though what have we gained from that, well now each token has an updated representation that incorporates information from itself and earlier tokens in the prompt, pretty much building context for the prompt and we see how that all worked with the mathematical computations behind attention.

Now as you can see, in the architecture diagram, there is a second RMSNorm, which is pretty easy for us because we’ve outputted an output matrix W_o, or you can think of it as the output vectors for each of the tokens after the residual add and so we can just normalize them to prevent blow up or zeroing before we go to the Feed Forward Network (FFN).

Now the weights in the FFN are very important, they were maticulously modified during training to ensure that with the help of attention as well when we produced the next token and the ones after that it was a coherent correct answer.

Lets think through this with an analogy, think of a group of people working through problem, where where each person represents one token and holds a page of notes which is the token’s vector.
Attention is like exchanging information between the people,
remember though not everyone can share due to causal masking, you
can only share amongst the people before you.

    FFN is like after you have exchanged information with whoever you

can for one round, everyone is given a paper of rules or hints
that they apply to their answers individually without consulting
anyone that further enriches this answer.

Within our model architecture, this process repeats 16 times between transformer layers.

FFN pretty much is just adding more information to our output vectors, independently, without the context of the other tokens within the prompt. It is looking at the features within the token, and using learned weights to build the representation better.

I hope that little intro made a bit of sense, but now we are going to go into the thing that really builds what a feed forward network really is. The perceptron, you can think of a perceptron as being one weighted decision, This means there are weights, and there are inputs and we apply certain weights to certain input values to compute a mathematical value that could bring us to some conclusion. This is best understood in an example.
Imagine a program that tries to decide whether a message is urgent. We give it two numbers:
x[0] = 1 if the message contains “urgent”; otherwise 0
x[1] = the number of exclamation marks

And then we have a code representation of a perceptron:
float score =
1.5f _ x[0] +
0.25f _ x[1] -
1.0f;

bool is_urgent = score > 0.0f;

You can see the weights we applied, we put a more significant importance to the number of times the message contains urgent as opposed to the number of exclamation marks, and this makes sense to us as the goal is quite simple and its easy to say that a message containing the word urgent would garner a higher score for checking urgency. The -1 is a bias. It shifts the point where the answer changes.

This describes the basic shape of a perceptron; take input numbers, calculate a weighted sum, and apply a rule to produce an output.
Now we did this as a basic example for 2 weights and 2 inputs but for n inputs it’d be more like this:

score = weight[0] × input[0] + weight[1] × input[1] + … + weight[n-1] × input[n-1] + bias
This sum evidently mirrors the dot product computation.
Now imagine doing that same weighted sum calculation thousands of times using the same input vector where each calculation has its own learned weights, which means we’re taking the same input numbers and combining them differently each time.
We can put all these sets of weights into a matrix, where each row holds the weights for one calculation as we’ve done before. Then multiplying that matrix by our input vector gives us all the weighted sums together. Now each row gives us one number, and those numbers together make up our new output vector.
This is pretty much what we mean by a linear projection, in Llama 3.2 1B model an FFN expansion projection takes the token’s 2048 values and produces 8192 values so we now have a larger vector containing different learned combinations of the input features, which gives the network more room to process them.
Remember though, in our urgency example we took the weighted sum and checked whether it was above zero to get a yes or no. Llama’s FFN keeps numerical values and uses a form of nonlinear processing called SwiGLU.
So now we know how we can take a token’s 2048 values, multiply them by a learned weight matrix and get 8192 values back. In Llama’s FFN we actually do this twice, once using a matrix called W_up and again using a separate matrix called W_gate. Remember though both of these are working with the same input vector, which is the token’s vector after our second RMSNorm, meaning we aren’t doing the up projection and then passing its output into the gate projection. They both start from the same input, they just have different learned weights.
Let’s call that normalized input vector x, then for one token we have:
x = 2048 values
W_up = [8192, 2048]
W_gate = [8192, 2048]
As we talked about before, every row contains one set of weights that we use to calculate one weighted sum, so having 8192 rows means we get 8192 output values. Each row has 2048 weights because we need one weight for each of the 2048 input values.
The actual computations look like this:
up = x × W_upᵀ
gate = x × W_gateᵀ
We transpose the weight matrices here because we’re treating x as a row vector, so the dimensions become:
[1, 2048] × [2048, 8192] = [1, 8192]

up[0] =
x[0] × W_up[0, 0]

- x[1] × W_up[0, 1]
- ...
- x[2047] × W_up[0, 2047]
  Then for up[1] we use the next row of weights, but still the exact same input vector, and we keep going until we’ve calculated all 8192 values. The gate projection does the same thing using its own weights. Also remember our urgency perceptron had a bias, but these Llama projections don’t, so here we just calculate the weighted sum.
  The up projection is called up because we’re expanding from 2048 values to 8192, pretty much giving the network more intermediate features to work with. This doesn’t mean we’re copying the original values or just adding zeros to make the vector bigger, each new value comes from a learned combination of the input values. Meaning we now have a larger space to process these different combinations before bringing the vector back down to its original size. Just like with our hidden states though, we can’t assume every position has a neat English meaning that we can point to.
  The gate projection also gives us 8192 values, but we use these to control how the values from the up projection contribute to the next computation. One thing to remember here is that the gate’s weights are fixed during inference, but its output values aren’t, because those values depend on the token vector we put in. So the same learned matrix can produce different gate values for different tokens.
  Before we combine the two vectors, we apply an activation function called SiLU to each value in the gate vector. An activation function is pretty much a function we apply to the result of a weighted sum, like in our urgency example where we checked whether the score was greater than zero to give us a boolean. With SiLU we keep a numerical value, and the calculation is:
  SiLU(g) = g × sigmoid(g)
  sigmoid(g) = 1 / (1 + e^(-g))
  SiLU(g) = g / (1 + e^(-g))
  Where g is one value from our gate vector. The sigmoid part gives us a number between 0 and 1, then we multiply that by the original gate value.
  Let’s go through an example, say one of our gate values is 2:
  sigmoid(2) = 1 / (1 + e^(-2)) ≈ 0.8808
  SiLU(2) = 2 × 0.8808 ≈ 1.7616
  sigmoid(-2) = 1 / (1 + e²) ≈ 0.1192
  SiLU(-2) = -2 × 0.1192 ≈ -0.2384
  sigmoid(0) = 0.5
  SiLU(0) = 0 × 0.5 = 0
  So you can see that a positive value can pass through fairly strongly, whereas a very negative value gets pushed close to zero. It’s also a smooth function, so during training we can calculate gradients through it to help adjust the weights.
  Now that we’ve applied SiLU to the gate vector, we multiply the result by our up vector element by element. Meaning we take the first value from each vector and multiply them, then the second value from each, and keep doing that for all 8192 values.
  intermediate[j] = SiLU(gate[j]) × up[j]
  Remember this isn’t a dot product, we don’t add the results together, so we still have 8192 values afterwards.
  Let’s use smaller vectors so we can actually see the whole computation:
  gate = [2, -2, 0]
  up = [3, 4, 5]
  SiLU(gate) ≈ [1.7616, -0.2384, 0]

intermediate[0] = 1.7616 × 3 ≈ 5.2848
intermediate[1] = -0.2384 × 4 ≈ -0.9536
intermediate[2] = 0 × 5 = 0
intermediate ≈ [5.2848, -0.9536, 0]
This is where the idea of a gate comes from, we have one branch producing values and another branch producing factors that scale those values based on the input. Looking at the example, the first up value got amplified, the second changed sign and became smaller in magnitude, and the third became zero. So don’t think of this as a strict on/off gate, and don’t assume the multiplier is always between 0 and 1 either. That’s only true for the sigmoid part, the full SiLU output can be negative or greater than 1.
This combination of SiLU and the elementwise multiplication is called SwiGLU. Pretty much, it lets the contribution of a feature from the up branch depend on a feature calculated by the gate branch, with both coming from the same token’s input vector. These relationships are learned during training.
The nonlinear computation is important here because if we only multiplied by one learned matrix to expand the vector, then another learned matrix to bring it back down, we could combine those matrices into one and get the same result. So adding SiLU and multiplying the two branches gives us more ways to process the input than a single linear projection could.
Now we’ve got our processed vector of 8192 values, but remember the token’s existing hidden state has 2048 values, and we eventually need to add our update back to it through the residual connection. So we use another learned matrix called W_down:
W_down = [2048, 8192]
down = intermediate × W_downᵀ
The dimensions are:
[1, 8192] × [8192, 2048] = [1, 2048]

down[0] =
intermediate[0] × W_down[0, 0]

- intermediate[1] × W_down[0, 1]
- ...
- intermediate[8191] × W_down[0, 8191]
  Then we do this for every row until we have all 2048 output values. Bringing the vector down doesn’t mean we just delete the extra values, we’re using learned weights to mix the 8192 intermediate values into a smaller vector.
  Lastly we add this update back to the token’s existing hidden state through the residual connection. One detail here, we add it to the hidden state from before the second RMSNorm, because the normalized vector was what we used inside the FFN branch. So we need to keep that original hidden state available for the addition.
  Putting the whole computation together:
  x = RMSNorm(hidden_state)
  up = x × W_upᵀ
  gate = x × W_gateᵀ
  intermediate = SiLU(gate) ⊙ up
  down = intermediate × W_downᵀ
  new_hidden_state = hidden_state + down
  Where ⊙ just means the elementwise multiplication we went through above.
  Where ⊙ just means the elementwise multiplication we went through above.
  We’ve been looking at one token to make it easier to follow, but for the whole prompt we can put the token vectors into rows of a matrix and calculate them together. So the expansion projections go from [token_count, 2048] to [token_count, 8192], then the down projection returns us to [token_count, 2048]. Every token position uses the same weights within this layer, but each row is processed independently, meaning we aren’t mixing information between token positions during these FFN computations.
  So now we’ve gone through all the computations inside one transformer layer, but remember from our architecture diagram that our Llama 3.2 1B model has 16 of these layers, meaning what we’ve just finished is layer 0, and we still have layers 1 through 15 to go through.
  The way this works is pretty much taking the hidden states we got after the FFN’s residual add and passing them into the next layer, where we do the same sequence of operations again. We don’t go back to the embedding table and start over, the updated vectors from the previous layer become the input for the next one. So when layer 1 calculates its queries, keys and values, it’s calculating them from representations that have already been processed by layer 0.
  Remember though, having the same operations doesn’t mean every layer uses the same weights. Each layer has its own learned Q, K, V and output projection matrices, its own RMSNorm weights, and its own gate, up and down projection matrices. Meaning the FFN weights are shared between token positions within one layer, like we mentioned before, but they’re different from one layer to another. So each layer gets to work with what the earlier layers have already computed, attention can combine information between token positions again, then the FFN processes the resulting features within each position. We aren’t manually assigning each layer a specific job, their different weights were learned together during training to make this whole sequence useful for predicting the next token.
  One thing to keep in mind is that for our prompt, the hidden-state matrix still has the same shape after every complete layer, [token_count, 2048]. The numbers inside it have changed, but each token still has a 2048-value vector, which lets us pass it straight into the next layer. These are still internal representations at this point, we haven’t picked an output token after each layer.
  Then as per the architecture diagram we add the final RMSNorm before an output.

So now we’ve gone through all 16 transformer layers and applied the final RMSNorm, but remember what we have at this point is still a vector of 2048 numbers for a token, these numbers contain the information the model has processed but they aren’t directly telling us what token should come next. For that we need the LM head, where LM pretty much means language modeling, and the head is the part that turns our internal representation or vector into scores/probabilities for possible output tokens.
Remember our model has a vocabulary of 128256 tokens, so we need to go from 2048 hidden-state values to 128256 scores, one for every possible token ID. We do this using another learned weight matrix:
W_lm_head = [128256, 2048]
Each row belongs to one vocabulary token and contains 2048 learned weights, meaning we can take the dot product of our final normalized vector with each row to get a score for that token. These weights were learned during training along with the rest of the model, so the earlier layers and the LM head have been trained together to make useful next-token predictions.
For generating the next token, we use the vector from the last position in the prompt. Remember causal masking means that position can gather information from all the positions before it as well as itself, so by the time it reaches the end of the model its vector can contain information from the whole prompt.
Each position’s output is used to predict what comes after that position, meaning if we want to continue the prompt, it’s the last position’s prediction we need. Our engine therefore only runs this final normalization and LM head calculation on the last prompt position.
Let’s call that final normalized vector h:
h = [1, 2048]
logits = h × W_lm_headᵀ
The dimensions become:
[1, 2048] × [2048, 128256] = [1, 128256]
And looking at just the score for a vocabulary token with ID v:
logits[v] = h[0] × W_lm_head[v, 0] + h[1] × W_lm_head[v, 1] + … + h[2047] × W_lm_head[v, 2047]
So it’s the same weighted sum computation we’ve already seen, just now each result has a direct connection to a vocabulary token. There’s no bias in this projection either.
This also connects back to why we spent so much time updating the hidden states through attention and the FFN, changing those hidden-state values changes the dot products we calculate here, meaning it changes which vocabulary tokens receive higher or lower scores.
The scores we get are called logits. They can be positive or negative and they aren’t probabilities yet, so a score of 5 doesn’t mean a 50% chance or anything like that. What we can say is that a higher logit gives that token a higher ranking for this particular next-token prediction.
Now we need to actually choose a token, and in our engine we use argmax. This pretty much means finding the position of the largest value in the logits vector. One thing to keep in mind is that we want the index of that value, because the index is the token ID, the score itself isn’t what we send to the tokenizer.
Let’s use a pretend vocabulary with only five tokens so we can see this clearly:
Token ID: 0 1 2 3 4
Logit: 0.4 -1.1 3.7 2.0 -0.3
The largest score is 3.7, but our chosen token ID is 2, because that’s the position where we found the largest score.
max(logits) = 3.7
argmax(logits) = 2
For our actual model we do the same thing across all 128256 scores. The basic idea can be written as:
next_token_id = 0
highest_score = logits[0]

for each remaining token ID:
if logits[token_id] > highest_score:
highest_score = logits[token_id]
next_token_id = token_id

This way of choosing the highest-scoring token is called greedy decoding. For the same logits it picks the same highest-scoring token each time, we aren’t randomly sampling from the possible outputs.
You may remember softmax from attention, and we can also apply it here if we want to convert the logits into probabilities:
probability[v] = exp(logits[v]) / sum(exp(logits))
Where the sum includes every vocabulary token. But for argmax we don’t actually need to calculate this, because softmax preserves the ordering of the scores, meaning whichever token had the highest logit will also have the highest probability. So our engine can just find the largest logit directly.
Lastly, the token ID we chose needs to be decoded using the tokenizer so we can get its text or bytes. Remember a token could be a whole word, part of a word, punctuation or something else, so choosing one token doesn’t necessarily mean we’ve generated a whole word.
That chosen token becomes the next position in the sequence, and when generation continues, the model processes that new position and uses its final vector to predict the token after it.
