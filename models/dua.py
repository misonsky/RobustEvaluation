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
class AttentionFlow(keras.Model):
    def __init__(self,hidden_size):
        super(AttentionFlow, self).__init__()
        self.encoder=Encoder(enc_units=hidden_size)
    def build(self,input_shape):
        hidden_size=input_shape[-1]
        self.w1=self.add_weight(name="v1", 
                                shape=[hidden_size,hidden_size], 
                                dtype=tf.float32,
                                initializer=tf.keras.initializers.GlorotNormal(),
                                trainable=True)
        self.w2=self.add_weight(name="v2", 
                                shape=[hidden_size,hidden_size], 
                                dtype=tf.float32,
                                initializer=tf.keras.initializers.GlorotNormal(),
                                trainable=True)
        self.bias=self.add_weight(name="b1", 
                                shape=[hidden_size],
                                dtype=tf.float32,
                                initializer=tf.keras.initializers.GlorotNormal(),
                                trainable=True)
        self.v3=self.add_weight(name="v3", 
                                shape=[hidden_size], 
                                dtype=tf.float32,
                                initializer=tf.keras.initializers.GlorotNormal(),
                                trainable=True)
    def single_utterance(self,utterane):
        """
        parameter:
            utterance : batch * seq_len *d 
        """
        left=tf.einsum("bsd,dt->bst",utterane,self.w1)
        right=tf.einsum("bsd,dt->bst",utterane,self.w2)
        tile_left=tf.tile(tf.expand_dims(left,axis=2),multiples=[1,1,utterane.shape[1],1])
        temp_rep=tf.tanh(tile_left + tf.expand_dims(right,axis=1) + self.bias) # b * m* n *d
        score=tf.nn.softmax(tf.einsum("bmnd,d->bmn",temp_rep,self.v3),axis=-1)
        c=tf.einsum("bmn,bnd->bmd",score,utterane)
        p,_=self.encoder(tf.concat([utterane,c],axis=-1))
        return p
        
        
    def call(self,features):
        """
        parameters:
            fetures: b *(turn +1) * seq_len *d
        """
        features_list=tf.unstack(features,axis=1)
        p_list=[]
        for utter in tf.unstack(features_list):
            """
            parameter:
                utter:b * seq_len * d
            """
            p_utter=self.single_utterance(utter)
            p_list.append(p_utter)
        return tf.stack(p_list,axis=1)
class TurnAware(keras.Model):
    def __init__(self):
        super(TurnAware, self).__init__()
    def call(self,features):
        """
            parameters:
                features: b * (turn +1) * seq_len *d
            return b * (turn +1) * seq_len * 2d 
        """
        fusion_feature=list()
        list_feature=tf.unstack(features,axis=1)
        last_utterence=list_feature[-2]
        for utterance in list_feature:
            fusiont_utter=tf.concat([utterance,last_utterence],axis=-1)
            fusion_feature.append(fusiont_utter)
        return tf.stack(fusion_feature,axis=1)
class Aggregation(keras.Model): 
    def __init__(self,filters,kernel_size,strides,pool_kernel_size,pool_strides):
        super(Aggregation, self).__init__()
        self.conv_3d=keras.layers.Conv3D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding="same",
                                          activation="elu",
                                          use_bias=True,
                                          kernel_initializer=keras.initializers.glorot_normal(),
                                          bias_initializer=keras.initializers.glorot_normal())
        self.pool_3d=keras.layers.MaxPool3D(pool_size=pool_kernel_size,
                                            strides=pool_strides,
                                            padding="same")
    def build(self,input_shape):
        self.input_channel=input_shape[-1]
    def call(self,features):
        """
            parameters:
                features:batch * num_turn * seq_len_u * seq_len_r * d
        """
        conv_reult=self.conv_3d(features)
        return self.pool_3d(conv_reult)
class MatchReponse(keras.Model):
    def __init__(self):
        super(MatchReponse, self).__init__()
    def build(self, input_shape):
        self.kernel=self.add_weight(name="A",shape=[input_shape[-1],input_shape[-1]]) 
    def wordMatch(self,features):
        """
            paramters:
                features: b * (turn +1) * seq_len * d
        """
        turn_list=tf.unstack(features,axis=1)
        response=turn_list[-1]
        utterance=tf.stack(turn_list[:-1],axis=1)
        word_m=tf.einsum("btmk,bnk->btmn",utterance,response)
        return word_m
    def utterMatch(self,features):
        """
        parameter:
            features:b * (turn +1) seq_len * d
        """
        turn_list=tf.unstack(features,axis=1)
        response=turn_list[-1]
        utterance=tf.stack(turn_list[:-1],axis=1) # b * turn * seq_len * d
        left_dot=tf.einsum("btmd,dk->btmk",utterance,self.kernel)
        utter_m=tf.einsum("btmd,bnd->btmn",left_dot,response)
        return utter_m
    def call(self,WordM,UttterM):
        word_m=self.wordMatch(WordM)
        utter_m=self.utterMatch(UttterM)
        return word_m,utter_m
        
class DUA(keras.Model):
    def __init__(self, vocab_size, embedding_matrix,config):
        super(DUA, self).__init__()
        self.batch_sz = config.batch_size
        self.enc_units = config.hidden_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                                   trainable=True)
        self.encoder1=Encoder(enc_units=config.hidden_size)
        self.encoder2=Encoder(enc_units=config.hidden_size)
        self.turn_awear=TurnAware()
        self.attentionF=AttentionFlow(config.hidden_size)
        self.match_response=MatchReponse()
        self.aggregat1=Aggregation(filters=config.filter_size[0],
                                   kernel_size=(3,3,3),
                                   strides=(1,1,1),
                                   pool_kernel_size=(3,3,3),
                                   pool_strides=(3,3,3))
        self.aggregat2=Aggregation(filters=config.filter_size[1],
                                   kernel_size=(3,3,3),
                                   strides=(1,1,1),
                                   pool_kernel_size=(3,3,3),
                                   pool_strides=(3,3,3))
        self.flatten=keras.layers.Flatten()
        self.output_prj=keras.layers.Dense(2)  
    @tf.function
    def call(self,history,response):
        his=self.embedding(history)
        res=self.embedding(response)
        # encoder
        h_r,_=self.encoder1(res)
        h_u_list=[]
        for utter in tf.unstack(his,axis=1):
            h_u,_=self.encoder2(utter)
            h_u_list.append(h_u)
        # construct the S list 
        h_u_list.append(h_r)
        awear_represent=self.turn_awear(tf.stack(h_u_list,axis=1))
        p_ret=self.attentionF(awear_represent)
        word_m,utter_m=self.match_response(tf.stack(h_u_list,axis=1),p_ret)
        conv_result=self.aggregat2(self.aggregat1(tf.stack([word_m,utter_m],axis=-1)))
        final_info=self.flatten(conv_result)
        logits=self.output_prj(final_info)
        y_pre=tf.nn.softmax(logits,axis=-1)
        return logits,y_pre
        
        
        
        
        
        
        
            
        
        
        
        
        
            
        
        
        
        
        
        
        