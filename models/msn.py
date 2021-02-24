#coding=utf-8
import tensorflow as tf
from tensorflow  import keras
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
class TransformerBlock(keras.Model):
    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm=keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear1=keras.layers.Dense(input_size)
        self.linear2=keras.layers.Dense(input_size)
    def FFN(self, X):
        return self.linear2(tf.nn.relu(self.linear1(X)))
    def call(self,Q, K, V, episilon=1e-8):
        '''
        parameters:
            Q: (batch_size, max_r_words, embedding_dim)
            K: (batch_size, max_u_words, embedding_dim)
            V: (batch_size, max_u_words, embedding_dim)
            return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        '''
        d_model=Q.shape[-1]
        Q_K=tf.einsum("brd,bud->bru",Q,K) /tf.math.sqrt(tf.cast(d_model+episilon,dtype=tf.float32))
        Q_K_score = tf.nn.softmax(Q_K,axis=-1)
        V_att=tf.einsum("bru,bud->brd",Q_K_score,V)
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output
class Attention(keras.Model):
    def __init__(self,hidden_size):
        super(Attention, self).__init__()
        self.linear1=keras.layers.Dense(hidden_size)
        self.linear1=keras.layers.Dense(1)
    def call(self,X):
        '''
        parameters:
         X:
        mask:   http://juditacs.github.io/2018/12/27/masked-attention.html  
        '''
        M = tf.nn.tanh(self.linear1(X))  # (batch_size, max_u_words, embedding_dim)
        M = self.linear2(M)
        score = tf.nn.softmax(M, axis=1)    # (batch_size, max_u_words, 1)
        output=tf.reduce_sum(score * X,axis=1)
        return output
class MSN(keras.Model):
    def __init__(self, vocab_size, embedding_matrix,config):
        super(MSN, self).__init__()
        self.batch_sz = config.batch_size
        self.enc_units = config.hidden_size
        self.config=config
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                                   trainable=True)
        self.alpha = 0.5
        self.gamma = 0.3
        self.selector_transformer = TransformerBlock(input_size=config.hidden_size)
        self.W_word=self.add_weight(name="W_word", 
                               shape=[config.hidden_size,config.hidden_size,config.max_turn], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        self.v=self.add_weight(name="v", 
                               shape=[config.max_turn,1], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        self.linear_word=keras.layers.Dense(1)
        self.linear_score=keras.layers.Dense(1)
        self.transformer_utt = TransformerBlock(input_size=config.hidden_size)
        self.transformer_res = TransformerBlock(input_size=config.hidden_size)
        self.transformer_ur = TransformerBlock(input_size=config.hidden_size)
        self.transformer_ru = TransformerBlock(input_size=config.hidden_size)
        self.A1=self.add_weight(name="A1", 
                               shape=[config.hidden_size,config.hidden_size], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        self.A2=self.add_weight(name="A2", 
                               shape=[config.hidden_size,config.hidden_size], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        self.A3=self.add_weight(name="A3", 
                               shape=[config.hidden_size,config.hidden_size], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        self.conv1=keras.layers.Conv2D(filters=config.msn_filter[0],
                                      kernel_size=(3,3),
                                      strides=(1,1),
                                      activation="relu",
                                      kernel_initializer=keras.initializers.he_normal(),
                                      padding="same")
        self.max_pooling1=keras.layers.MaxPool2D(pool_size=(2,2),
                                                strides=(2,2),
                                                padding="same")
        self.conv2=keras.layers.Conv2D(filters=config.msn_filter[1],
                                      kernel_size=(3,3),
                                      strides=(1,1),
                                      activation="relu",
                                      kernel_initializer=keras.initializers.he_normal(),
                                      padding="same")
        self.max_pooling2=keras.layers.MaxPool2D(pool_size=(2,2),
                                                strides=(2,2),
                                                padding="same")
        self.conv3=keras.layers.Conv2D(filters=config.msn_filter[2],
                                      kernel_size=(3,3),
                                      strides=(1,1),
                                      activation="relu",
                                      kernel_initializer=keras.initializers.he_normal(),
                                      padding="same")
        self.max_pooling3=keras.layers.MaxPool2D(pool_size=(3,3),
                                                strides=(3,3),
                                                padding="same")
        self.affine2=keras.layers.Dense(config.hidden_size)
        self.gru_acc=Encoder(enc_units=config.hidden_size)
        self.encoder1=Encoder(enc_units=config.hidden_size)
        self.encoder2=Encoder(enc_units=config.hidden_size)
        self.affine_out=keras.layers.Dense(2)
        self.dropout = keras.layers.Dropout(0.2)
    def word_selector(self, key, context):
        """
        parameters:
            key:  (bsz, max_u_words, d)
            context:  (bsz, max_turn,max_u_words, d)
            return: score
        """
        A=tf.einsum("blud,dth->bluth",context,self.W_word)
        A=tf.einsum("bluht,bmh->blumt",A,key) / tf.math.sqrt(tf.cast(key.shape[-1],dtype=tf.float32))
        A=tf.reshape(tf.einsum("blumd,dt->blumt",A,self.v), shape=[A.shape[0],A.shape[1],A.shape[2],A.shape[3]])
        a=tf.concat([tf.reduce_max(A,axis=2),tf.reduce_max(A,axis=2)], axis=-1)
        s1=tf.nn.softmax(tf.reshape(self.linear_word(a),shape=[a.shape[0],a.shape[1]]),axis=-1)
        return s1
    def utterance_selector(self, key, context):
        """
        parameters:
        key:  (bsz, max_u_words, d)
        context:  (bsz, max_turn,max_u_words, d)
        return: score:
        """
        key=tf.reduce_mean(key,axis=1)
        context = tf.reduce_mean(context,axis=2)
        s2=tf.einsum("bud,bd->bu",context,key)/(1e-6 + tf.norm(context,ord=2,axis=-1)*tf.norm(key,ord=2,axis=-1,keepdims=True))
        return s2
    def distance(self, A, B, C, epsilon=1e-6):
        M1=tf.einsum("but,brt->bur",tf.einsum("bud,dt->but",A,B),C)
        A_norm = tf.norm(A,ord=2,axis=-1)
        C_norm = tf.norm(C,ord=2,axis=-1)
        M2=tf.einsum("bud,brd->bur",A,C) /(tf.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
        return M1, M2
    def context_selector(self,context, hop=[1, 2, 3]):
        """
        parameters:
            context: (batch_size, max_utterances, max_u_words, embedding_dim)
            key: (batch_size, max_u_words, embedding_dim)
        """
        su1, su2, su3, su4 = context.shape
        context_=tf.reshape(context, shape=[-1,su3, su4])
        context_ = self.selector_transformer(context_, context_, context_)
        context_=tf.reshape(context_,shape=[su1, su2, su3, su4])
        multi_match_score = []
        for hop_i in hop:
            key = tf.reduce_mean(context[:,self.config.max_turn-hop_i:, :, :],axis=1)
            key = self.selector_transformer(key, key, key)
            s1 = self.word_selector(key, context_)
            s2 = self.utterance_selector(key, context_)
            s = self.alpha * s1 + (1 - self.alpha) * s2
            multi_match_score.append(s)
        multi_match_score = tf.stack(multi_match_score, axis=-1)
        match_score = tf.reshape(self.linear_score(multi_match_score),shape=[multi_match_score.shape[0],multi_match_score.shape[1]])
        mask = (tf.nn.sigmoid(match_score) >= self.gamma)
        mask = tf.cast(mask,dtype=tf.float32)
        match_score = match_score * mask
        context = context *tf.expand_dims(tf.expand_dims(match_score, axis=-1),axis=-1)
        return context
    def get_Matching_Map(self, bU_embedding, bR_embedding):
        """
            bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
            bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
            return: E: (bsz*max_utterances, max_u_words, max_r_words)
        """
        M1, M2 = self.distance(bU_embedding, self.A1, bR_embedding)
        Hu = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
        Hr = self.transformer_res(bR_embedding, bR_embedding, bR_embedding)
        M3, M4 = self.distance(Hu, self.A2, Hr)
        Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)
        Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
        M5, M6 = self.distance(Hur, self.A3, Hru)
        M = tf.stack([M1, M2, M3, M4, M5, M6], axis=1)
        return M
    def UR_Matching(self, bU_embedding, bR_embedding):
        """
            bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
            bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
            :return: (bsz*max_utterances, (max_u_words - width)/stride + 1, (max_r_words -height)/stride + 1, channel)
        """
        M = self.get_Matching_Map(bU_embedding, bR_embedding)

        Z = tf.nn.relu(self.conv1(M))
        Z = self.max_pooling1(Z)

        Z = tf.nn.relu(self.conv2(Z))
        Z =self.max_pooling2(Z)

        Z = tf.nn.relu(self.conv3(Z))
        Z =self.max_pooling3(Z)
        Z =tf.reshape(Z, shape=[Z.shape[0],-1])
        V = tf.nn.tanh(self.affine2(Z))   # (bsz*max_utterances, 50)
        return V
    @tf.function
    def call(self, bU, bR):
        """
        bU: batch utterance, size: (batch_size, max_utterances, max_u_words)
        bR: batch responses, size: (batch_size, max_r_words)
        return: scores, size: (batch_size, )
        """
        bU_embedding = self.embedding(bU) # + self.position_embedding(bU_pos) # * u_mask
        bR_embedding = self.embedding(bR) # + self.position_embedding(bR_pos) # * r_mask
#         bR_embedding,_= self.encoder1(bR_embedding)
#         turn_utter=[]
#         for utter in tf.unstack(bU_embedding,axis=1):
#             u_r,_=self.encoder2(utter)
#             turn_utter.append(u_r)
#         bU_embedding=tf.stack(turn_utter,axis=1)
        multi_context = self.context_selector(bU_embedding, hop=[1, 2, 3,4,5])
        su1, su2, su3, su4 = multi_context.shape
        multi_context = tf.reshape(multi_context,shape=[-1, su3, su4])
        sr1, sr2, sr3= bR_embedding.shape
        bR_embedding = tf.tile(tf.expand_dims(bR_embedding, axis=1),multiples=[1, su2, 1, 1])
        bR_embedding = tf.reshape(bR_embedding,shape=[-1, sr2, sr3])
        V = self.UR_Matching(multi_context, bR_embedding)
        V = tf.reshape(V,shape=[su1, su2, -1])
        H, _ = self.gru_acc(V)
        L = self.dropout(H[:,-1,:])
        logits=self.affine_out(L)
        y_pre=tf.nn.softmax(logits,axis=-1)
        return logits,y_pre
        
    
            
        
        
        
        
        

        
        
        
        
    