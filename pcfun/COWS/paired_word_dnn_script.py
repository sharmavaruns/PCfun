"""
Train a DNN classifier for paired words using pre-trained vectors.
"""
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F

#%%
class runDNN_supervised(nn.Module):
    """
    Binary classification of paired words using pre-trained word vectors.
    """
    def __init__(
        self, X, units=[500],act_fns = [nn.ReLU()],out_layers_n_nodes = 1,
        dropout=.5,epochs=100,batch_size=128, trainable_vectors=False
    ):
        """
        Initialize a PairedDNNClassifier.
        Args:
             - X (pd.DataFrame): vector pairs features in pd.DataFrame format.
             - units (list): list of units of the DNN. The length of the
                list determine the number of layers. Defaults to [64, 16].
            - dropout (float): dropout rate. Defaults to 0.5.
            - trainable_vectors (bool): trainable vectors. Defaults to True.
        """
        super(runDNN_supervised, self).__init__()
        # create embedding with the pretrained vectors.
        self.embedding_layer = create_embedding_layer(
            X.values, not trainable_vectors
        )
        # add to the hidden units the first layer for the paired words.
        ## note: X.shape[1] corresponds to number of features, hence 1 neuron in input layer
        ## corresponding to 1 feature in dataset
        # add to the hidden units the first layer for the paired words.
        self.units = [X.shape[1]] + units
        self.number_of_layers = len(units)
        self.dropout = dropout
        self.stacked_dense_layers = nn.Sequential(*[
            create_dense_layer(input_size, output_size,dropout=dropout,
                              activation_fn=act_fn)
            for input_size, output_size, act_fn in zip(self.units, self.units[1:],act_fns)
        ])
        # add the binary classification layer
        self.output = create_dense_layer(
            self.units[-1], out_layers_n_nodes,
            activation_fn=nn.Sigmoid(),
            dropout=0.0
        )
        self.epochs = epochs
        self.batch_size = batch_size
        
    def forward(self, X):
        """
        Apply the forward pass of the model.
        Args:
            - pair: a torch.Tensor containing the indexes of the
                paired words.
        Returns:
            a torch.Tensor with the score for the pairs.
        """
        #embedded_pair =  self.embedding_layer(torch.LongTensor([i for i in range(X.shape[0])]))
        #encoded_pair = self.stacked_dense_layers(embedded_pair)
        encoded_pair = self.stacked_dense_layers(X)
        return self.output(encoded_pair)
    def fit(self,X_train,y_train,loss = nn.BCELoss(),binary_y = True):
        """
        This function does the training of the DNN when input an X_train and y_train pd.DataFrames
        Args:
            - X_train: pd.DataFrame with the training features
            - y_train: pd.Series/pd.DataFrame with the training labels
        Output:
            DNN fit (classifier) object
        """
        # binary classification so binary cross entropy loss
        self.loss = loss
        self.opt = optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999)
        )
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.train()
            losses_train = []
            accuracies_train = []
            labels_train_list = []
            pred_labels_train = []
            samples_train = torch.FloatTensor(X_train.values)
            if binary_y:
                labels_train = torch.FloatTensor([[i] for i in y_train.values])
            else:
                labels_train = torch.FloatTensor(y_train.values)
            #print(samples_train.shape,labels_train.shape)
            for beg_i in range(0, samples_train.shape[0], self.batch_size):
                x_batch = samples_train[beg_i:beg_i + self.batch_size, :]
                y_batch = labels_train[beg_i:beg_i + self.batch_size, :]
                #print(x_batch.shape,y_batch.shape)
                x_batch = Variable(x_batch)
                self.x_batch = x_batch
                y_batch = Variable(y_batch)
                self.y_batch = y_batch
                self.opt.zero_grad()
                # (1) Forward
                y_hat = self.forward(x_batch)
                # (2) Compute diff
                loss = self.loss(y_hat, y_batch)
                if binary_y:
                    accuracies_train.append(
                        (
                            (y_hat >.5).numpy().flatten() ==
                            y_batch.numpy().flatten()
                        ).astype(int).mean()
                    )
                    labels_train_list.append(y_batch)
                    pred_labels_train.append(y_hat > 0.5)
                else:
                    from sklearn.metrics.pairwise import cosine_similarity as cosine
                    accuracies_train.append(F.cosine_similarity(y_hat,y_batch,1).mean())    
                    #accuracies_train.append(1)
                # (3) Compute gradients
                loss.backward()
                # (4) update weights
                self.opt.step()        
                losses_train.append(loss.data.numpy())
                self.losses_train = losses_train
            if binary_y:
                print('epoch={}\tloss_train={}\taccuracy_train={}'.format(epoch+1,
                                                                          sum(losses_train)/float(len(losses_train)),
                                                                          sum(accuracies_train)/len(accuracies_train)
                                                                         )
                     )
            else:
                self.accuracies_train = accuracies_train
                print('epoch={}\tloss_train (distance)={}\tcosine_sims={}'.format(epoch+1,
                                                                                  sum(losses_train)/float(len(losses_train)),
                                                                                  sum(accuracies_train)/len(accuracies_train)
                                                                                  
                                                                 )
                     )
        return(self)
    def predict_proba(self,X_test,binary_y = True):
        # evaluate the model
        self.eval()
        self.prediction = self.forward(torch.FloatTensor(X_test.values))
        if binary_y == True:
            prediction = np.array([vals[0] for vals in self.prediction.tolist()])
        else:
            prediction = self.prediction
            #print(prediction.shape)
        return(prediction)

class runDNN_supervised_old(nn.Module):
    def __init__(self,X,y,units=[500],dropout=0.5,epochs=100,batch_size=128,trainable_vectors=False):
        '''
        something to describe this class here
        '''
        ## Make attributes from nn.Module available within this class environment
        super(runDNN_supervised_old, self).__init__()
        self.embedding_layer = create_embedding_layer(
            X.values, not trainable_vectors
        )
        # add to the hidden units the first layer for the paired words.
        ## note: X.shape[1] corresponds to number of features, hence 1 neuron in input layer
        ## corresponding to 1 feature in dataset
        self.units = [X.shape[1]] + units
        self.number_of_layers = len(units)
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.stacked_dense_layers = nn.Sequential(*[
            create_dense_layer(input_size, output_size,dropout=dropout)
            for input_size, output_size in zip(self.units, self.units[1:])
        ])
        # add the binary classification layer
        self.output = create_dense_layer(
            self.units[-1], 1,
            activation_fn=nn.Sigmoid(),
            dropout=0.0
        )
        # binary classification so binary cross entropy loss
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999)
        )

    def forward(self,X):
        """
        Apply the forward pass of the model.
        Args:
            - pair: a torch.Tensor containing the indexes of the
                paired words.
        Returns:
            a torch.Tensor with the score for the pairs.
        """
        embedded_pair =  self.embedding_layer(torch.LongTensor([i for i in range(X.shape[0])]))
        encoded_pair = self.stacked_dense_layers(embedded_pair)
        return self.output(encoded_pair)
    def fit(self,X_train,y_train):
        """
        This function does the training of the DNN when input an X_train and y_train pd.DataFrames
        Args:
            - X_train: pd.DataFrame with the training features
            - y_train: pd.Series/pd.DataFrame with the training labels
        Output:
            DNN fit (classifier) object
        """
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.train()
            y_train[y_train == -1] = 0
            y_train_torch = torch.FloatTensor([[y_val] for y_val in y_train])
            losses_train = []
            accuracies_train = []
            labels_train = []
            pred_labels_train = []
            #for samples, labels in train_dataloader:
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.forward(X_train)
            #print(y_train_torch.shape)
            loss = self.criterion(outputs, y_train_torch)
            accuracies_train.append(
                    (
                        (outputs >.5).numpy().flatten() ==
                        y_train_torch.numpy().flatten()
                 ).astype(int).mean()
                )
            labels_train.append(y_train_torch)
            pred_labels_train.append(outputs > 0.5)
            loss.backward()
            self.optimizer.step()
            # collect loss
            losses_train.append(loss.item())
            print('epoch={}\naccuracy_train={}\tloss_train={}'.format(epoch + 1,
                                                                      sum(accuracies_train) / len(accuracies_train),
                                                                      sum(losses_train) / float(len(losses_train))
                                                                     )
                 )
        return(self)
    def predict_proba(self,X_test):
        # evaluate the model
        self.eval()
        self.prediction = self.forward(X_test)
        prediction = [vals[0] for vals in self.prediction.tolist()]
        return(np.array(prediction))

class Vectors_Pairs():
    '''
    Args:
    - df: full DF with PCn>>>>GOn (x,1001) dimensions
    Output:
    - pairs: with the PCn, GOn pairs with their associated label
    - vectors: vector embedding for each 'phrase' (e.g. PC1,GO1, etc) (500 dimensional)
    '''
    def __init__(self,df,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'
                 )
                ):
        ## n_dimensions of the original vectors: 500 dims in this case
        n_dims = int((df.shape[1]-1)/2)
        pcs_names = [name.split('>>>>')[0] for name in df.index]
        gos_names = [name.split('>>>>')[1] for name in df.index]
        pcs_df = copy.deepcopy(df.iloc[:,0:n_dims])
        pcs_df.index = pcs_names
        pcs_df.columns = range(n_dims)
        gos_df = copy.deepcopy(df.iloc[:,n_dims:1000])
        gos_df.index = gos_names
        gos_df.columns = range(n_dims)

        idxs = pd.concat([pcs_df,gos_df]).index
        vectors = pd.concat([pcs_df,gos_df]).iloc[[not val for val in idxs.duplicated()],:]
        pairs = pd.DataFrame({'first':pcs_names,'second': gos_names,'label': df['label']})
        pairs.index = range(len(pcs_names))
        # REMEMBER TO CHANGE THIS LINE ONCE MATTEO GETS BACK TO YOU
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # change negative labels from -1 to 0 for DNN
        pairs.loc[pairs['label'] == -1, 'label'] = 0
        'Add the pairs and word_to_index dictionary to the class object'
        self.pairs = pairs
        self.word_to_index = {
            word: index
            for index, word in enumerate(
                vectors.index.tolist()
            )
        }
        self.vectors = vectors
    def __len__(self):
        """
        Get number of pairs.
        
        Returns:
            the number of pairs.
        """
        return len(self.pairs)
    def __getitem__(self, index):
        """
        Get a pair and the associated label.
        Args:
            - index (int): the index of a pair.
        Returns:
            a tuple with two torch.Tenors:
                - the first containing the pair indexes.
                - the second containing the label.
        """
        row = self.pairs.iloc[index]
        pair = torch.from_numpy(
            np.array([
                [self.word_to_index[row['first']]],
                [self.word_to_index[row['second']]]
            ])
        ).to(device=self.device)
        label = torch.tensor(
            [row['label']],
            dtype=torch.float, device=self.device
        )
        return pair, label

class Korn(Dataset):
    def __init__(self,vectors):
        self.vectors = vectors
    def __getitem__(self, index):
        """
        Get a pair and the associated label.
        Args:
            - index (int): the index of a pair.
        Returns:
            a tuple with two torch.Tenors:
                - the first containing the pair indexes.
                - the second containing the label.
        """
        row = self.vectors.iloc[index]
        return torch.tensor(row)
    
class WordPairsDataset(Dataset):
    """Word pairs dataset."""

    def __init__(
        self, pairs, vectors,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        """
        Initialize the WordPairsDataset.
        Args:
            filepath (string): path to the csv file with the pairs.
            vectors_filepath (string): path to the csv file with the vectors.
                Used to map the words to indexes.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        #if not all([isinstance(input_df,pd.core.frame.DataFrame) for input_df in [pairs,vectors]]):
        #    raise ValueError('input pairs and vectors must be a pd.DataFrame')
        # REMEMBER TO CHANGE THIS LINE ONCE MATTEO GETS BACK TO YOU
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # change negative labels from -1 to 0 for DNN
        if -1 in set(pairs['label']):
            pairs.loc[pairs['label'] == -1, 'label'] = 0
        self.pairs = pairs
        self.word_to_index = {
            word: index
            for index, word in enumerate(
                vectors.index.tolist()
            )
        }
        self.device = device

    def __len__(self):
        """
        Get number of pairs.
        
        Returns:
            the number of pairs.
        """
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Get a pair and the associated label.
        Args:
            - index (int): the index of a pair.
        Returns:
            a tuple with two torch.Tenors:
                - the first containing the pair indexes.
                - the second containing the label.
        """
        row = self.pairs.iloc[index]
        pair = torch.from_numpy(
            np.array([
                [self.word_to_index[row['first']]],
                [self.word_to_index[row['second']]]
            ])
        ).to(device=self.device)
        label = torch.tensor(
            [row['label']],
            dtype=torch.float, device=self.device
        )
        return pair, label

#%%
def create_dense_layer(
    input_size, output_size,
    activation_fn=nn.ReLU(), dropout=.5
):
    """
    Create a dense layer.
    Args:
        - input_size (int): size of the input.
        - output_size (int): size of the output.
        - activation_fn (an activation): activation function.
            Defaults to ReLU.
        - dropout: dropout rate. Defaults to 0.5.
    Returns:
        a nn.Sequential.
    """
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(input_size, output_size)),
        ('activation_fn', activation_fn),
        ('dropout', nn.Dropout(p=dropout)),
]))


def create_embedding_layer(vectors, non_trainable=False):
    """
    Create an embedding layer.
    Args:
        - vectors (np.ndarray): word vectors in numpy format.
        - non_trainable (bool): non trainable vectors. Defaults to False.
    Returns:
        a nn.Embedding layer.
    """
    number_of_vectors, vector_dimension = vectors.shape
    embedding_layer = nn.Embedding(number_of_vectors, vector_dimension)
    embedding_layer.load_state_dict({'weight': torch.from_numpy(vectors)})
    if non_trainable:
        embedding_layer.weight.requires_grad = False
    return embedding_layer


class PairedWordsDNNClassifier(nn.Module):
    """
    Binary classification of paired words using pre-trained word vectors.
    """

    def __init__(
        self, vectors, units=[64, 16],
        dropout=.5, trainable_vectors=True
    ):
        """
        Initialize a PairedDNNClassifier.
        Args:
             - vectors (np.ndarray): word vectors in numpy format.
             - units (list): list of units of the DNN. The length of the
                list determine the number of layers. Defaults to [64, 16].
            - dropout (float): dropout rate. Defaults to 0.5.
            - trainable_vectors (bool): trainable vectors. Defaults to True.
        """
        super(PairedWordsDNNClassifier, self).__init__()
        # create embedding with the pretrained vectors.
        self.embedding_layer = create_embedding_layer(
            vectors, not trainable_vectors
        )
        # add to the hidden units the first layer for the paired words.
        self.units = [2*vectors.shape[1]] + units
        self.number_of_layers = len(units)
        self.dropout = dropout
        self.stacked_dense_layers = nn.Sequential(*[
            create_dense_layer(input_size, output_size,dropout=dropout)
            for input_size, output_size in zip(self.units, self.units[1:])
        ])
        # add the binary classification layer
        self.output = create_dense_layer(
            self.units[-1], 1,
            activation_fn=nn.Sigmoid(),
            dropout=0.0
        )
        
    def forward(self, pair):
        """
        Apply the forward pass of the model.
        Args:
            - pair: a torch.Tensor containing the indexes of the
                paired words.
        Returns:
            a torch.Tensor with the score for the pairs.
        """
        embedded_pair =  self.embedding_layer(pair).view(-1, self.units[0])
        encoded_pair = self.stacked_dense_layers(embedded_pair)
        return self.output(encoded_pair)