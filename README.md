# Transformers

### Language Modeling Overview
Language modeling is a task where a machine learning model is trained to understand and generate human language. It does this by learning patterns in natural language through probability. One of the most common ways of achieving this is through **next-token prediction**. A popular real-world example is autocorrect, where the model uses the earlier part of a sentence (the previous tokens) as context to predict the next token.

For example, given the context *“I was brushing the”*, the model is more likely to predict words like *“dog”* or *“cat”*, since these have higher probabilities learned from training data. It is very unlikely to predict words like *“banana”* or *“monitor”*.

To train a model like this, we can provide it with a real sentence such as:  
*“My favorite Overwatch character is Reinhardt.”*  
We give the model all tokens except the final one (*“Reinhardt”*) and ask it to predict that final token. This process is repeated at every position in the sentence by shifting the tokens. A loss function such as **cross-entropy loss** is then used to update the model’s parameters. After being trained on millions of such examples, the model learns meaningful probability distributions over language and performs well.

---

### Self-Attention and Transformers
Self-attention is a core component of Transformers and is a major reason for their success. Its purpose is to allow each token in a sentence to look at and weigh the importance of all other tokens in the same sentence. In natural language, words rarely exist independently.

For example, in the sentence:  
*“The cat tried to catch the mouse, but it got away.”*  
The model should infer that *“it”* refers to the mouse rather than the cat.

Self-attention works by creating three vectors for each token:
- **Query (Q):** represents what the current token is looking for  
- **Key (K):** represents how each token relates to others  
- **Value (V):** contains the actual content or information of each token  

Attention scores are computed using the dot product of the Query and Key vectors, followed by a softmax operation. These scores are then used to weight the Value vectors, allowing the model to focus on the most relevant parts of the sentence.

---

### Model Architecture
To implement the Transformer model, I used three main classes.

#### Transformer Class
The `Transformer` class is responsible for managing the overall flow of the model, from token input to the final output logits. It initializes:
- Token embeddings
- Sinusoidal positional encodings
- A list of `Block` modules
- A final layer normalization
- A linear output layer that projects the final token representations (of size `d_model`) to `vocab_size`

#### Block Class
The `Block` class is the main repeating unit of the Transformer. My model uses **6 blocks (layers)**. Each block refines token representations using two sub-layers:
1. **Masked self-attention**
2. **Feed-forward network**

The feed-forward network processes each token independently after it has gathered contextual information from the attention layer. This is where the model learns more complex language patterns.

---

### Hyperparameters and Training Decisions
I designed my model with:
- **128-dimensional embeddings**
- **6 layers**
- **8 attention heads**
- **512 feed-forward hidden units**

Initially, I experimented with much larger dimensions, but training took around 45 minutes per run. After trial and error, I found that these smaller dimensions produced similar validation results while being significantly more efficient.

I also experimented extensively with learning rate and dropout:
- Early on, a low dropout rate caused overfitting. This was evident because the model stopped improving after about 6 epochs, even though training continued for 15 epochs.
- A dropout rate between **0.15 and 0.2** worked best.
- I tested learning rates of **0.03**, **0.003**, and **0.0003**, and achieved the best results with **0.0003**.

---

### Results
After extensive experimentation, the following results were obtained from the best-performing model:

| Metric      | Train  | Validation | Test  |
|------------|--------|------------|-------|
| Perplexity | 26.59  | 47.46      | 42.45 |
| Loss       | 3,991,854.62 | 371,311.86 | 408,592.73 |
