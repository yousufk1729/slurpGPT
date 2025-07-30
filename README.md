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

I also added a top-k scorer which feeds 100 Shakespeare lines as context into the model, which then predicts the k most likely tokens to appear after. After each guess, the correct token is added to the context and the model guesses again. This is my half-asses solution to the model not knowing what a "word" is (i.e. predicting the next word, which is a normal task for something like LAMBADA, is not something my model is set up to do). Implementation is based on my GPT generator, and I do want to clean up this class. The top-5 accuracy seems to be around 65%, and I have no idea if this is a good number. The issue is that my tokenizer vocabulary is so little that I still have character-level tokens, which causes words that don't exist to be generated. Of course, my model also hasn't trained for long with lots of parameters.

## Sample 1: Richard III

**Context:**
```
I cry thee mercy:
There is my purse to cure that blow of thine.
Hath any well-advised friend proclaim'd
Reward to him that brings the traitor in?
Third Messenger:
Such proclamation hath been made, my liege.
Fourth Messenger:
Sir Thomas Lovel and Lord Marquis Dorset,
'Tis said, my liege, in Yorks
```

**Target:** ` hire are in arm`

### Predictions

| Step | Actual Token | Top-5 Predictions |
|------|--------------|-------------------|
| **1** | `'hi'` (372) | 1. `','` (44) - **32.8%** <br> 2. `'hi'` (372) - **32.5%** ✅ <br> 3. `'.'` (46) - **20.8%** <br> 4. `':'` (58) - **7.1%** <br> 5. `' and'` (296) - **6.8%** |
| **2** | `'re'` (264) | 1. `'re'` (264) - **80.4%** ✅ <br> 2. `'p'` (112) - **18.2%** <br> 3. `'ps'` (942) - **0.6%** <br> 4. `'m'` (109) - **0.4%** <br> 5. `'e'` (101) - **0.3%** |
| **3** | `' are'` (418) | 1. `','` (44) - **47.3%** <br> 2. `' is'` (324) - **17.0%** <br> 3. `"'s"` (320) - **12.5%** <br> 4. `':'` (58) - **12.4%** <br> 5. `'!'` (33) - **10.8%** |
| **4** | `' in'` (307) | 1. `' in'` (307) - **28.7%** ✅ <br> 2. `' f'` (271) - **22.7%** <br> 3. `' made'` (752) - **18.5%** <br> 4. `' with'` (336) - **16.4%** <br> 5. `' al'` (665) - **13.6%** |
| **5** | `' arm'` (901) | 1. `' arm'` (901) - **78.4%** ✅ <br> 2. `' the'` (267) - **6.7%** <br> 3. `' our'` (412) - **5.6%** <br> 4. `' his'` (347) - **4.7%** <br> 5. `' hand'` (635) - **4.5%** |

## Sample 2: Romeo & Juliet

**Context:**
```
I'll have this knot knit up to-morrow morning.
JULIET:
I met the youthful lord at Laurence' cell;
And gave him what becomed love I might,
Not step o'er the bounds of modesty.
CAPULET:
Why, I am glad on't; this is well: stand up:
This is as't should be. Let me see the county;
Ay, marry, go, I say
```

**Target:** `, and fetch`

### Predictions

| Step | Actual Token | Top-5 Predictions |
|------|--------------|-------------------|
| **1** | `','` (44) | 1. `','` (44) - **67.4%** ✅ <br> 2. `" '"` (438) - **14.2%** <br> 3. `'.'` (46) - **9.7%** <br> 4. `';'` (59) - **4.9%** <br> 5. `' I'` (291) - **3.8%** |
| **2** | `' and'` (296) | 1. `' I'` (291) - **39.4%** <br> 2. `' go'` (482) - **21.6%** <br> 3. `' and'` (296) - **14.9%** ✅ <br> 4. `' be'` (304) - **14.3%** <br> 5. `' to'` (287) - **9.7%** |
| **3** | `' f'` (271) | 1. `' I'` (291) - **43.2%** <br> 2. `' let'` (537) - **17.0%** <br> 3. `' tell'` (701) - **14.4%** <br> 4. `' you'` (288) - **14.1%** <br> 5. `','` (44) - **11.4%** |
| **4** | `'et'` (314) | 1. `'et'` (314) - **32.8%** ✅ <br> 2. `'ind'` (501) - **20.0%** <br> 3. `'oo'` (332) - **16.6%** <br> 4. `'ull'` (838) - **15.7%** <br> 5. `'ell'` (408) - **14.9%** |
| **5** | `'ch'` (322) | 1. `'ch'` (322) - **97.8%** ✅ <br> 2. `'ter'` (404) - **1.7%** <br> 3. `'che'` (998) - **0.3%** <br> 4. `'he'` (257) - **0.1%** <br> 5. `'ite'` (888) - **0.0%** |

This is quite fun to play around with and helps elucidate the idea of all LLMs just being functions that determine probabilites of subsequent tokens. 

## Next Steps
Based on the training vs. validation loss, the model is quite overfit, which is a consequence of the model not actually doing anything with the validation data to improve hyperparameters. I added early stopping before it got too bad:

<img width="1920" height="967" alt="training" src="https://github.com/user-attachments/assets/6529c77c-12a3-484e-9436-854f78225ccd" />

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
