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

A safetensor file is made of 3 sections, header size, header, tensor data. A quick aside; a tensor is a multi-dimensional array or numbers, where tensordata here represents the model weights.

Header size is always 8 bytes, each unsigned 64 bit, The header is in JSON meaning its just a group of key value pairs. Where the key is a unique tensor name. Every value is a JSON itself containing 3 keys; dtype, shape, offsets. Dtype says what data type tensor is stored in. shape says the dimensions of the tensor, as i mentioned above a tensor is a multidimensional array. Offsets say where the tensor is stored within the tensors data section. It is a start and end to show where the tensors raw numbers are stored in the files data section.

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
