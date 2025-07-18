# slurpGPT
GPT implementation using only PyTorch to generate text similar to a training example. 

## Motivation
The transformer architecture is incredibly important and it’s time for us to implement it. We will only use PyTorch instead of using higher-level libraries (tiktokenizer, sentencepiece, langchain, huggingface) and API wrappers. 

## Architecture
This project has the following files:
- tokenizer.py: Custom tokenizer using GPT-4 regex pattern + BPE algorithm. Hyperparameter for vocab size. 
- gpt.py: Custom GPT using typical decoder-only transformer architecture (multi-headed self-attention/feedforward blocks w/ layer normalization/residual connections/dropout). 
- train_tokenizer.py: Trains the tokenizer. 
- train_gpt.py: Trains the model. Has most hyperparameters.
- generate.py: Generates either random text or text from an input prompt, where the model will continue where you left off. Only 1 parameter for amount of tokens to generate. 

This implementation is based primarily on Andrej Karpathy’s deep learning and tokenizer tutorials and the seminal paper *Attention Is All You Need*. I think these resources do a better job of explaining theory. 

## Training/Results
The input, compiled by Andrej Karpathy, contains a collection of Shakespeare’s scripts (seems to be various acts/scenes from *Coriolanus*, *Richard III*, *Romeo and Juliet*, *Henry VI*, *The Winter’s Tale*, *Measure for Measure*, *The Taming of the Shrew*, *The Tempest*). After training for almost two hours (!), the model achieves some OK results, acting as a *Lorem ipsum* generator with Shakespearean flavour. For example:

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

Considering that this model effectively only predicts the next character, it’s interesting how it is able to form scripts that look coherent until you actually start reading them. It reminds me of those “English for Non-English Speaker” YouTube videos. 

*Edit: I have changed the model without retraining, so currently the "strongest" version of the GPT which I use as an example above does not exist on the latest commit.*

## Next Steps
Based on the training vs. validation loss, the model is strongly overfit, even with dropout:

![Graph](https://github.com/user-attachments/assets/25f5b94c-45a8-4c69-affe-6952550d7a30)

The solution is to have more input text, as the model is already quite large with 10,788,929 total parameters. I want to add my favourites King Lear and Macbeth to the mix. 

Given that the training accuracy was continuing to improve, more iterations would also help improve training accuracy, but not the overfitting problem. 

There is also potential for hyperparameter optimization using some third-party libraries or just manual testing. 

## Attempted Next Steps
I tried using GELU to copy OpenAI (seems like GELU is falling out of favour compared to SwiGLU though...). I found that ReLU, which was used in *Attention Is All You Need*, was actually giving better results. 

## References
- Andrej Karpathy’s GPT implementation: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
- Andrej Karpathy's tokenizer implementation: https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
- Attention is All You Need: https://arxiv.org/pdf/1706.03762   
- Dropout: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf 
- ResNet: https://arxiv.org/pdf/1512.03385 
