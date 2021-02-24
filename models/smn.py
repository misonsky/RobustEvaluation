#coding=utf-8
import tensorflow as tf
from tensorflow import keras

class Encoder(keras.Model):
    def __init__(self,enc_units):
        super(Encoder, self).__init__()
        self.gru = tf.keras.layers.GRU(enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer=keras.initializers.glorot_normal())
    def call(self, x):
        whole_states, final_state = self.gru(x)
        return whole_states, final_state

class Token2TokenMatch(keras.Model):
    def __init__(self):
        super(Token2TokenMatch,self).__init__()
    def call(self,utterance,response):
        """
        note: this layer is different the orign paper
                here using the hidden state of GRU
        parameters:
            utterance:b * turn * u *d
            response: b * r *d
        return:
            b * turn * u * r
        """
        M=tf.einsum("btud,bdr->btur",utterance,tf.transpose(response,perm=[0,2,1]))
        return M
class Seq2SeqMatch(keras.Model):
    def __init__(self):
        super(Seq2SeqMatch,self).__init__()
    def build(self, input_shape):
        self.kernel=self.add_weight(name="A",shape=[input_shape[-1],input_shape[-1]])
    def call(self,utterance,response):
        """
        parameters:
            utterance:b * turn * u *d
            response:b * r * d
        """
        left_result=tf.einsum("btud,dm->btum",utterance,self.kernel)
        right_result=tf.einsum("btud,bdr->btur",left_result,tf.transpose(response,perm=[0,2,1]))
        return right_result

class Accumulation(keras.Model):
    def __init__(self,max_turn,hidden_size=50,filter_size=8,kernel_size=(3, 3)):
        super(Accumulation,self).__init__()
        self.max_turn=max_turn
        self.conv=keras.layers.Conv2D(filters=filter_size,
                                      kernel_size=kernel_size,
                                      activation=keras.activations.relu,
                                      kernel_initializer=keras.initializers.he_normal(),
                                      padding="valid")
        self.max_pooling=keras.layers.MaxPool2D(pool_size=kernel_size,
                                                strides=kernel_size,
                                                padding="valid")
        self.flatten=keras.layers.Flatten()
        self.dense_prj=keras.layers.Dense(units=hidden_size,
                                      activation=keras.activations.tanh,
                                      kernel_regularizer=keras.initializers.GlorotNormal())
    def call(self,matrix1,matrix2):
        turns=[]
        matrix=tf.stack([matrix1,matrix2],axis=4) # b * turn * u * r *2
        Turn_Rep=tf.unstack(matrix, num=self.max_turn, axis=1)
        for single_turn in Turn_Rep:
            conv_result=self.conv(single_turn)
            max_pool=self.max_pooling(conv_result)
            matching_vector=self.dense_prj(self.flatten(max_pool))
            turns.append(matching_vector)
        matching_vectors=tf.stack(turns,axis=1)
        return matching_vectors
class SMN(keras.Model):
    def __init__(self, vocab_size, embedding_matrix,config):
        super(SMN, self).__init__()
        self.batch_sz = config.batch_size
        self.enc_units = config.hidden_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                                   trainable=True)
        self.encoder1=Encoder(enc_units=config.hidden_size)
        self.encoder2=Encoder(enc_units=config.hidden_size)
        self.token_match=Token2TokenMatch()
        self.seq_match=Seq2SeqMatch()
        self.accumulation=Accumulation(max_turn=config.max_turn)
        self.output_prj=keras.layers.Dense(2)
    @tf.function
    def call(self,history,response):
        his=self.embedding(history)
        res=self.embedding(response)
        singles_turn=tf.unstack(his,axis=1)
        combine=list()
        for turn in singles_turn:
            history_rep,_=self.encoder1(turn)
            combine.append(history_rep)
        history_rep=tf.stack(combine,axis=1)
        response_rep,_=self.encoder1(res)
        Token_S=self.token_match(history_rep,response_rep)# b * turn * u * r
        Seq_s=self.seq_match(history_rep,response_rep)# b * turn * u * r
#         matrix=tf.stack([Token_S,Seq_s],axis=4) # b * turn * u * r *2
        match_vectors=self.accumulation(Token_S,Seq_s)
        _,final_state=self.encoder2(match_vectors)
        logits=self.output_prj(final_state)
        y_pre=tf.nn.softmax(logits,axis=-1)
        return logits,y_pre
        
        
        
        
        
        
        
    
