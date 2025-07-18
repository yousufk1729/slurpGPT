# slurpGPT
GPT implementation using only PyTorch to generate text similar to a training example. 

## Motivation
The transformer architecture is incredibly important and it’s time for us to implement it. We will only use PyTorch instead of using higher-level libraries (tiktokenizer, sentencepiece, langchain, huggingface) and API wrappers. 

## Architecture
This project has the following files:
- tokenizer.py: Custom tokenizer using GPT-4 regex pattern + BPE algorithm. Hyperparameter for vocab size. 
- gpt.py: Custom GPT using typical decoder-only transformer architecture (multi-headed self-attention/feedforward blocks w/ layer normalization/residual connections/dropout). Has a modest 11,526,400 parameters.
- train_tokenizer.py: Trains the tokenizer. 
- train_gpt.py: Trains the model. Has most hyperparameters.
- generate.py: Generates either random text or text from an input prompt, where the model will continue where you left off. Only 1 parameter for amount of tokens to generate. 

This implementation is based primarily on Andrej Karpathy’s deep learning and tokenizer tutorials and the seminal paper *Attention Is All You Need*. I think these resources do a better job of explaining theory. 

## Training/Results
I used [this](https://www.kaggle.com/datasets/kewagbln/shakespeareonline?resource=download) Shakespeare dataset, which contains Shakespeare’s First Folio (36 plays) + *Pericles, Prince of Tyre*, *The Sonnets*, and *A Lover's Complaint*. However, I didn’t examine it hard enough and I found that it copied the same paragraph of license 218 times along with lots of excess whitespace. After 1 hour and 45 minutes of training on my GTX 1650, the model achieves O.K results, acting as a *Lorem ipsum* generator with Shakespearean flavour. For example:

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

Considering that this model effectively only predicts the next few characters, it’s interesting how it is able to form scripts that look coherent until you actually start reading them. It reminds me of those “English for Non-English Speaker” YouTube videos. 

*The above example was generated with an earlier commit, I forgot to copy-paste new examples but the quality is still about the same.*

## Next Steps
Based on the training vs. validation loss, the model is strongly overfit. To be fair, I set the dropout percentage to be zero:

<img width="863" height="538" alt="image" src="https://github.com/user-attachments/assets/bbec329c-75fd-444f-bb47-c1fb762dc54a" />

Given that the training accuracy was continuing to improve, more iterations would also help improve the model. I am not really concerned with overfitting because quite frankly, overfitting data to all of Shakespeare is kind of the goal here. 

I also want to reformat the input training data to have less whitespace and remove the license copies. *After formatting with some regex, the new file is 4844KB and the old one was 5321KB.*

There is also potential for hyperparameter optimization using some third-party libraries or just manual testing. 

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
