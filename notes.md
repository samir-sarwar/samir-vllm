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
