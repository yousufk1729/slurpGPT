# slurpGPT
11M-parameter GPT implementation using only PyTorch to generate text similar to a training example. 

### *Project is basically finished aside from training. I got the educational value I intended to get from this project (first time using Python and PyTorch).*

## Motivation

The transformer architecture is incredibly important and it’s time for us to implement it. We will only use PyTorch instead of using higher-level libraries (tiktokenizer, sentencepiece, langchain, huggingface) or creating an API wrapper (hackathon-style).

## Tokenizer

The tokenizer is an unfaithful reproduction of OpenAI’s GPT-2 tokenizer; it uses the common byte-pair encoding algorithm (BPE) with a regex pattern to prevent unnecessary merges [1, section 2.2]. The implementation is quite simple: I used Karpathy’s code as a starting point and I copied the regex pattern directly from OpenAI’s tiktokenizer library [2] [3]. 

When training, I cut off the vocabulary size at 1000 to decrease computation time (3-4 minutes). I trained the tokenizer on a small corpus of Shakespeare’s plays, originally created by Karpathy [4]. This resulted in 416453 tokens for training (90% of total tokens, I saved 10% for validation). 

To play around with the full GPT-2 tokenization, I suggest an app I found online [5]. 

## GPT Architecture

The theory behind this architecture is primarily based on OpenAi’s GPT-2 paper, which is itself based on the GPT-1 paper and four other seminal papers: Attention Is All You Need, Layer Normalization, Deep Residual Learning for Image Recognition, and Dropout [1, section 2.3] [6] [7] [8] [9] [10]. I’m quite happy to see that the University of Toronto appears on a few of these papers because it makes my tuition cost feel slightly more justified. 

The implementation is based on Karpathy’s GPT implementations, which I highly recommend for any student [11] [12]. I was also frequently checking PyTorch documentation, which itself links to many of the above papers. 

If I had to summarize, I would describe the model as a simple decoder-only transformer with blocks containing multi-head attention and a feedforward network, with some tricks (layer norm, dropout, residuals w/ scaled weights, weight tying) to improve computation/avoid overfitting. The total model has 11122944 parameters.

Below is a flowchart with relevant details. The code is also quite heavily annotated. 

(I will eventually make this into a flowchart)

> (Input text) → Token embedding + positional embedding → N blocks →
>
> Each block: LayerNorm → Multi-Head Attention with Causal Masking → Dropout → Residual Connection -> LayerNorm → Feed Forward (single hidden layer with 4x nodes, GELU)→ Dropout → Residual Connection →
>
> Output of N blocks → layer norm → linear layer (de-embedding/output projection) → (Output text)

To match GPT-2:
- Weight tying input embedding weights with output projection weights
- GeLU in feedforward networks
- Scaled initialization for residual projection weights
- Pre-layer normalization along with additional layer normalization after all blocks

PyTorch-specific:
- Flash attention to make the scaled dot-products compute faster

## Training/Results
I used various weights for different training speeds, including weights recommended by Karpathy in his implementations [11] [12]. This model trained for around 2 hours on my laptop's GTX 1650, but a smaller version trains in 3 minutes with similar output quality. The model acts as a *Lorem ipsum* generator with Shakespearean flavour. For example:

> **KING HENRY VI:**  
> And, that, to came to the city,  
> All till'd of the brack, and bestroke  
> Have we spark'd with the chamber,  
> And broke the pursuries of their power,  
> And why dost thou shalt nothing end,  
> And, in the mind of thy father'stle,  
> And nothing end of thy father, to thy brothers,  
> Ere soonsing to the priest of thy father.  
>  
> **QUEEN MARGARET:**  
> Ah, my lord, and sovereign,  
> Even soonestiers of the king,  
> And bravereign, the king of York,  
> Have not soonest thou shalt not thy brother's son,  
> And why, and the bosom of thy smilkingle,  
> Ere in thy highness of York,  
> And thou shalt not soonest of the king,  
> And thou shalt nothing of thy brother's death,  
> And thou sharet not thy smilvers,  
> And brave thy brother's sorrow and thy face,  
> And make the duke of cause, and thy brokenessity,  
> And bid thy boasters and thy brother's sake,  
> And thou wilt not soont, and thou shalt not thy fault,  
> And foul miscument, thy brother,  
> And bold in thy father's face,  
> And thou shalt thou wilt not thy sleep'st my bosom,  
> And thou shalt nothing end thy brother's servereign's death,  
> And, in thy mischanced, thy horse,  
> And thou shalt not, thou liest thy brokeness,  
> And thou shalt not thy daughter'st thy fault to thy head;  
> And yet thou shalt thou shalt not to thee,  
> But that thou art not thy hour,  
> And thou wilt not thy father's purposeding,  
> And thou art not thy wilt thou not thy chamber,  
> And thou shalt not so best thou destroying to thee?  
>  
> **KING HENRY VI:**  
> What is the city?  
>  
> **QUEEN MARGARET:**  
> They that thou dost stay, thou art not?  


Here is some text from an older version:

> **KING RICHARD II:**  
> Ratcliff more fable than all proud;  
> The trumpets of wanton thousand men  
> That enforce the substance of the names:  
> The loss when treacherous ground fear of France blood,  
> Raise up the villain, because to die:  
> Marry, do am Lord Hastings, our hidding-clothed with kings,  
> To-day, lord, so much indeed, not with me.   
>  
> **ROMEO:**  
> Shall I not, so well; but I would be a king;  
> Bear thou a king, it moves to no world,  
> That I have provided myself most straightly,  
> Will take a lost with me to the wall.  
>  
> **FRIAR LAURENCE:**  
> My brother, would I were a husband for him  
> Romeo's grave; one now so he received,  
> One that ever was a son so deep as torment.  
> Romeo stands me, for more captives worn it.  
>  
> **ROMEO:**  
> Brave foot-page, peace! thou that thou art poor of Richmond,  
> Words thou rather possess'd thy mother's land.

Considering that this model effectively only predicts the next few characters, it’s interesting how it is able to form scripts that look coherent until you actually start reading them. It reminds me of those “What English Sounds Like To Non-English Speakers” YouTube videos. 

I also added a top-k scorer which feeds 100 Shakespeare lines as context into the model, which then predicts the k most likely tokens to appear after. 

(add results of that here)

## Next Steps
Based on the training vs. validation loss, the model is quite overfit. I added early stopping before it got too bad:

(img)

I do feel a bit limited by my computational resources; paying for a cloud GPU service is overkill for this educational exercise. I would like to try some more hyperparameter optimization techniques with the validation data and other methods of testing. 

## References (IEEE coming later)
- [1] https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf 
- [2] https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py  
- [3] https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py 
- [4] https://github.com/karpathy/ng-video-lecture/blob/master/input.txt 
- [5] https://tiktokenizer.vercel.app/?model=gpt2 
- [6] https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf 
- [7] https://arxiv.org/pdf/1706.03762 
- [8] https://arxiv.org/pdf/1607.06450 
- [9] https://arxiv.org/pdf/1512.03385 
- [10] https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf 
- [11] https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py 
- [12] https://github.com/karpathy/nanoGPT/blob/master/model.py 
