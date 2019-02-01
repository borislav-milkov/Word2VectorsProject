# Word2VectorsProject
The project uses a genism model to create word embeddings, it investigates each word in conjunction with the words around it and it determines it’s context. In such a way it is able to map words as vectors into (in this case 300) dimensional space. Words are clustered together based on meaning or relation in the corpus
## Model Training
The model trains with 50 epochs with the code provided. If you want to change the number of epochs look at the 'iter' variable in the model definition. 
## Saved Model
The model saves to the folder trained after training. This will allow you to skip training the next time and just jump to the already created word embeddings (vectors).
## Dimensionality
The embedding word vectors are 300 dimensional tensors. Since graphing in so many dimensions is impossible, we squash it to two dimensions using T-SNE. What T-SNE allows us to do is to reduce dimensionality while preserving our clustering of vectors (words). An intro to T-SNE can be seen here : https://www.youtube.com/watch?v=NEaUSP4YerM 

## Applications
You can find a lot of practical applications of word2vec. Here I am listing two of them.
1.	If you want to compare two sentences and see how similar they are, you can find the cosine similarity between those vectors.
2.	If have a document retrieval problem, you can compare the query vector with the document vectors using KL Divergence and retrieve relevant documents.

## Demo
We can see how our model clusters words which have similar vectors. We can see an example with food items being clustered. For more plots check out the Thrones2Vec iPython notebook. 

![capture](https://user-images.githubusercontent.com/25466748/52154449-bc3f5c00-2632-11e9-96af-c53a3cc13a7e.PNG)
