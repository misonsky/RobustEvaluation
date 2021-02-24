#coding=utf-8
from utils import positional_encoding
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
class AttentiveModule(keras.Model):
    def __init__(self):
        super(AttentiveModule,self).__init__()
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
    def build(self,input_shape):
        self.dense1=keras.layers.Dense(input_shape[-1],activation=keras.activations.relu)
        self.dense2=keras.layers.Dense(input_shape[-1])
    def dot_sim(self,x,y):
        assert x.shape[-1] == y.shape[-1]
        assert len(x.shape)==3 and len(y.shape)==3
        sim = tf.einsum('bik,bjk->bij', x, y)
        scale = tf.sqrt(tf.cast(x.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    def call(self,query,key,values):
        logits=self.dot_sim(query, key)
        attention = tf.nn.softmax(logits,axis=-1)
        assert len(query.shape) ==3 and len(key.shape) ==3
        weight_sum=tf.einsum('bij,bjk->bik', attention, values)
        result=self.layernorm1(weight_sum + query)
        ## FFN
        y=self.dense2(self.dense1(result))
        return self.layernorm2(y+result)
        
class SelfMatch(keras.Model):
    def __init__(self,num_layer):
        super(SelfMatch,self).__init__()
        self.num_layer = num_layer
        self.att=[AttentiveModule() for _ in tf.range(num_layer)]
    def call(self,_input):
        """
        parameter:
            his: batch * turn * seq_len * d
            res: batch * seq_len * d
        """
        stack_input=[_input]
        for i in range(self.num_layer):
            his=self.att[i](_input,_input,_input)
            stack_input.append(his)
        return stack_input
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
 
        
class DAM(keras.Model):
    def __init__(self, vocab_size,embedding_matrix,config):
        super(DAM, self).__init__()
        self.batch_sz = config.batch_size
        self.enc_units = config.hidden_size
        self.encoder1=Encoder(enc_units=config.hidden_size)
        self.encoder2=Encoder(enc_units=config.hidden_size)
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                                   trainable=True)
        self.pos_encoding = positional_encoding(config.max_utterance_len,config.hidden_size)
        self.match_self=SelfMatch(num_layer=config.num_layer)
        self.match_u_attend_r=AttentiveModule()
        self.match_r_attend_u=AttentiveModule()
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
        res,_=self.encoder1(res)
        combine_list=[]
        turn_list=tf.unstack(his,axis=1)
        for utterance in turn_list:
            utterance,_=self.encoder2(utterance)
            combine_list.append(utterance)
        Hr_stack = self.match_self(res)
        sim_turns=[]
        for utter in combine_list:
            # utter batch * seq_len * d
            Hu_stack=self.match_self(utter)
            u_a_r_stack,r_a_u_stack= [],[]
            for _index in range(len(Hr_stack)):
                u_a_r=self.match_u_attend_r(Hu_stack[_index],Hr_stack[_index],Hr_stack[_index])
                r_a_u=self.match_r_attend_u(Hr_stack[_index],Hu_stack[_index],Hu_stack[_index])
                u_a_r_stack.append(u_a_r)
                r_a_u_stack.append(r_a_u)
            u_a_r_stack.extend(Hu_stack)
            r_a_u_stack.extend(Hr_stack)
            u_a_r = tf.stack(u_a_r_stack, axis=-1)
            r_a_u = tf.stack(r_a_u_stack, axis=-1)
            sim = tf.einsum('biks,bjks->bijs', u_a_r, r_a_u) / tf.sqrt(tf.cast(self.enc_units,dtype=tf.float32))
            sim_turns.append(sim)
        sim = tf.stack(sim_turns, axis=1)
        conv_result=self.aggregat2(self.aggregat1(sim))
        final_info=self.flatten(conv_result)
        logits=self.output_prj(final_info)
        y_pre=tf.nn.softmax(logits,axis=-1)
        return logits,y_pre
        
        
        
        
        
        
        