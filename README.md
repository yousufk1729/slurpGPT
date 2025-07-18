# slurpGPT
GPT implementation using only PyTorch to generate text similar to a training example. 

## Motivation
The transformer architecture is incredibly important and it’s time for us to implement it. We will only use PyTorch instead of using higher-level libraries (tiktokenizer, sentencepiece, langchain, huggingface) and API wrappers. 

## Architecture
This project has the following files:
- tokenizer.py: Custom tokenizer using GPT-4 regex pattern + BPE algorithm. Hyperparameter for vocab size. 
- gpt.py: Custom GPT using typical decoder-only transformer architecture (multi-headed self-attention/feedforward blocks w/ layer normalization/residual connections/dropout). Older version had 10.8 million parameters, newer version has a modest 23,211,008 parameters.
- train_tokenizer.py: Trains the tokenizer. 
- train_gpt.py: Trains the model. Has most hyperparameters.
- generate.py: Generates either random text or text from an input prompt, where the model will continue where you left off. Only 1 parameter for amount of tokens to generate. 

This implementation is based primarily on Andrej Karpathy’s deep learning and tokenizer tutorials and the seminal paper *Attention Is All You Need*. I think these resources do a better job of explaining theory. 

## Training/Results
An earlier version of my model used character-level tokenization and [this](https://github.com/karpathy/ng-video-lecture/blob/master/input.txt) Shakespeare dataset compiled by Karpathy. It trained for around 2 hours on my laptop's GTX 1650. The model acts as a *Lorem ipsum* generator with Shakespearean flavour. For example:

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

A later version of my model used BPE tokenization with GPT-4 regex and [this](https://www.kaggle.com/datasets/kewagbln/shakespeareonline?resource=download) Shakespeare dataset, which contains Shakespeare’s First Folio (36 plays) + *Pericles, Prince of Tyre*, *The Sonnets*, and *A Lover's Complaint*. After 7 hour and 45 minutes (!) of training, the model achieves O.K results that were honestly worse that I was expecting:

> From fairest creatures we desire increase, and stomachs in
> 
> Concit the life to other. Other her beauteous left foils
> 
> Break the vary in the vessel of the night,
> 
> Stood 'twixt my consent and my fault
> 
> Those eyes of heaven; and of her favourites
> 
> Could make vileom do in them, ending them now
> 
> Such trades summon of her devour here
> 
> Into a full warlike person and his behalf
> 
> The envious statutes at his praise
> 
> To pluck mine eyes on you.

Considering that these models effectively only predict the next few characters, it’s interesting how they are able to form scripts that look coherent until you actually start reading them. It reminds me of those “English for Non-English Speaker” YouTube videos. 

## Next Steps
Based on the training vs. validation loss, the model is strongly overfit. To be fair, I set the dropout percentage to be zero. In hindsight, I shouldn't have done this and the first thing I would try is adding dropout back:

<img width="863" height="538" alt="image" src="https://github.com/user-attachments/assets/bbec329c-75fd-444f-bb47-c1fb762dc54a" />

I also think that Karpathy's input text was formatted better than the larger set I ended up using. 

There is also potential for hyperparameter optimization using some third-party libraries. I think at that point I would just throw in the towel and use tiktokenizer/sentencepiece + huggingface transformers though. 

*The above graph was generated with Claude Sonnet 4 because I was too lazy to format the output data.*

## Attempted Next Steps
I tried using GELU to copy OpenAI (seems like GELU is falling out of favour compared to SwiGLU though...). I found that ReLU, which was used in *Attention Is All You Need*, was actually giving better results. 

## References
- Andrej Karpathy’s GPT implementation: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
- Andrej Karpathy's tokenizer implementation: https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
- Shakespeare: https://www.kaggle.com/datasets/kewagbln/shakespeareonline?resource=download
- Attention is All You Need: https://arxiv.org/pdf/1706.03762   
- Dropout: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf 
- ResNet: https://arxiv.org/pdf/1512.03385 
