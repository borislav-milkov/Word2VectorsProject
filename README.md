# Word2VectorsProject
The project uses a genism model to create word embeddings, it investigates each word in conjunction with the words around it and it determines it’s context. In such a way it is able to map words as vectors into (in this case 300) dimensional space. Words are clustered together based on meaning or relation in the corpus
## Model Training
The model trains with 50 epochs with the code provided. If you want to change the number of epochs look at the 'iter' variable in the model definition. 
## Saved Model
The model saves to the folder trained after training. This will allow you to skip training the next time and just jump to the already created word embeddings (vectors).
## Dimensionality
The embedding word vectors are 300 dimensional tensors. Since graphing in so many dimensions is impossible, we squash it to two dimensions using T-SNE. What T-SNE allows us to do is to reduce dimensionality while preserving our clustering of vectors (words). An intor to T-SNE can be seen here : https://www.youtube.com/watch?v=NEaUSP4YerM 
