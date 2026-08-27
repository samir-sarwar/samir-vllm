Super rough notes on what I'm learning, can most definitley be incorrect things in here keep that in mind.

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

Header size is always 8 bytes, each unsigned 64 bit,
