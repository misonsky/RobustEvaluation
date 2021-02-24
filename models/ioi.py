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

class SelfAttention(keras.Model):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
    def build(self, input_shape):
        self.u=self.add_weight(name="u", 
                               shape=[input_shape[-1],input_shape[-1]], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        I = tf.eye(input_shape[-1])
        self.d=self.add_weight(name="d", 
                               shape=[input_shape[-1],input_shape[-1]], 
                               dtype=tf.float32, 
                               initializer=keras.initializers.GlorotNormal(), 
                               trainable=True)
        self.d=tf.multiply(self.d, I)
        self.dense1=keras.layers.Dense(input_shape[-1],activation=keras.activations.relu,name="att_dense1")
        self.dense2=keras.layers.Dense(input_shape[-1],name="att_dense1")
    def full_attention(self,utt_how, resp_how, dim=None):
        if dim==None:
            dim = utt_how.shape[-1]
        f1 = tf.nn.relu(tf.einsum('aij,jk->aik', utt_how, self.u), name='utt_how_relu') # [batch, len_utt, dim]
        f2 = tf.nn.relu(tf.einsum('aij,jk->aik', resp_how,self.u), name='resp_how_relu') # [batch, len_res, dim]
        S = tf.einsum('aij,jk->aik', f1, self.d)  # [batch, len_utt, dim]
        S = tf.einsum('aij,akj->aik', S, f2) # [batch, len_utt,len_res]
        return S
    def call(self,queries, keys,key_masks=None,query_masks=None,dropout_rate=0):
        """
        parameters:
            queries:A 3d tensor with shape of [N, T_q, C_q].
            keys:A 3d tensor with shape of [N, T_k, C_k].
            num_units: A scalar. Attention size.
        Returns:
            A 3d tensor with shape of (N, T_q, C)
        """
        hiddens =[]
        hiddens.append(queries)
        values = keys
        outputs = self.full_attention(queries, keys)
        scale = tf.maximum(1.0, keys.shape[-1] ** 0.5)
        outputs = outputs / scale
        if key_masks is None:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, axis=1), [1, queries.shape[1], 1])  # (N, T_q, T_k)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (N, T_q, T_k)
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        if query_masks is None:
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, keys.shape[1]])  # (N, T_q, T_k)
        outputs *= query_masks
        outputs=tf.nn.dropout(outputs, rate=dropout_rate)
        outputs = tf.matmul(outputs, values)  # ( h*N, T_q, C/h)
        outputs += queries
        outputs=self.layernorm1(outputs)
        outputs=self.dense2(self.dense1(outputs))
        hiddens.append(outputs)
        return hiddens
class IOI(keras.Model):
    def __init__(self, vocab_size, embedding_matrix,config):
        super(IOI, self).__init__()
        self.batch_sz = config.batch_size
        self.enc_units = config.hidden_size
        self.ioi_number = config.ioi_number
        self.config = config
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                                   trainable=True)
        self.context_self_attention=[SelfAttention() for _ in range(config.ioi_number)]
        self.response_self_attention=[SelfAttention() for _ in range(config.ioi_number)]
        self.context_cross_attention=[SelfAttention() for _ in range(config.ioi_number)]
        self.response_cross_attention=[SelfAttention() for _ in range(config.ioi_number)]
        self.conv1=[keras.layers.Conv2D(filters=config.filter_size[0],
                                      kernel_size=(3,3),
                                      strides=(1,1),
                                      activation="relu",
                                      kernel_initializer=keras.initializers.he_normal(),
                                      padding="same") for _ in range(config.ioi_number)]
        self.conv2=[keras.layers.Conv2D(filters=config.filter_size[1],
                                      kernel_size=(3,3),
                                      strides=(1,1),
                                      activation="relu",
                                      kernel_initializer=keras.initializers.he_normal(),
                                      padding="same") for _ in range(config.ioi_number)]
        self.pool1=[keras.layers.MaxPool2D(pool_size=(3, 3),
                                                strides=(3, 3),
                                                padding="same") for _ in range(config.ioi_number)]
        self.pool2=[keras.layers.MaxPool2D(pool_size=(3, 3),
                                                strides=(3, 3),
                                                padding="same") for _ in range(config.ioi_number)]
        self.dense1=[keras.layers.Dense(config.hidden_size,
                                      activation="relu",
                                      use_bias=True,
                                      kernel_regularizer=keras.initializers.GlorotNormal(),
                                      bias_initializer=keras.initializers.GlorotNormal(),name="dense1_%d"%(i)) for i in range(config.ioi_number)]
        self.dense2=[keras.layers.Dense(config.hidden_size,
                                      activation="relu",
                                      use_bias=True,
                                      kernel_regularizer=keras.initializers.GlorotNormal(),
                                      bias_initializer=keras.initializers.GlorotNormal(),name="dense2_%d"%(i)) for i in range(config.ioi_number)]
        self.dense3=[keras.layers.Dense(config.hidden_size,
                                      activation="relu",
                                      use_bias=True,
                                      kernel_regularizer=keras.initializers.GlorotNormal(),
                                      bias_initializer=keras.initializers.GlorotNormal(),name="dense3_%d"%(i)) for i in range(config.ioi_number)]
        self.dense4=[keras.layers.Dense(2,
                                      activation="relu",
                                      use_bias=True,
                                      kernel_regularizer=keras.initializers.GlorotNormal(),
                                      bias_initializer=keras.initializers.GlorotNormal(),name="dense4_%d"%(i)) for i in range(config.ioi_number)]
        self.encoder=[Encoder(config.hidden_size) for _ in range(config.ioi_number)]
        self.layer_morm1=keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_morm2=keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten=keras.layers.Flatten()
        
    @tf.function
    def call(self,history,response):
        context_mask = tf.cast(tf.not_equal(history, 0), tf.float32)
        response_mask = tf.cast(tf.not_equal(response, 0), tf.float32)
        context_embeddings=self.embedding(history)
        response_embeddings=self.embedding(response)
        expand_response_mask = tf.tile(tf.expand_dims(response_mask, 1), [1, self.config.max_turn, 1])
        expand_response_mask = tf.reshape(expand_response_mask, [-1, self.config.max_utterance_len]) 
        parall_context_mask = tf.reshape(context_mask, [-1, self.config.max_utterance_len])
        context_embeddings=tf.nn.dropout(context_embeddings,rate=self.config.dropout)
        response_embeddings=tf.nn.dropout(response_embeddings,rate=self.config.dropout)
        context_embeddings = tf.multiply(context_embeddings, tf.expand_dims(context_mask, axis=-1))  
        response_embeddings = tf.multiply(response_embeddings, tf.expand_dims(response_mask, axis=-1)) 
        expand_response_embeddings = tf.tile(tf.expand_dims(response_embeddings, 1), [1, self.config.max_turn, 1, 1])
        expand_response_embeddings = tf.reshape(expand_response_embeddings, [-1, self.config.max_utterance_len, self.config.emb_size])
        parall_context_embeddings = tf.reshape(context_embeddings, [-1, self.config.max_utterance_len, self.config.emb_size])
        context_rep, response_rep = parall_context_embeddings,expand_response_embeddings
        y_pred_list,logits_list= [],[]
        for k in range(self.config.ioi_number):
            inter_feat_collection = []
            context_self_rep=self.context_self_attention[k](queries=context_rep,
                                           keys=context_rep,
                                           key_masks=parall_context_mask,
                                           query_masks=parall_context_mask,
                                           dropout_rate=self.config.dropout)[1]
            response_self_rep=self.response_self_attention[k](queries=response_rep,
                                           keys=response_rep,
                                           key_masks=expand_response_mask,
                                           query_masks=expand_response_mask,
                                           dropout_rate=self.config.dropout)[1]
            context_cross_rep=self.context_cross_attention[k](queries=context_rep,
                                           keys=response_rep,
                                           key_masks=expand_response_mask,
                                           query_masks=parall_context_mask,
                                           dropout_rate=self.config.dropout)[1]
            response_cross_rep=self.response_cross_attention[k](queries=response_rep,
                                           keys=context_rep,
                                           key_masks=parall_context_mask,
                                           query_masks=expand_response_mask,
                                           dropout_rate=self.config.dropout)[1]
            context_inter_feat_multi = tf.multiply(context_rep, context_cross_rep)
            response_inter_feat_multi = tf.multiply(response_rep, response_cross_rep)
            context_concat_rep = tf.concat([context_rep, context_self_rep, context_cross_rep, context_inter_feat_multi], axis=-1) 
            response_concat_rep = tf.concat([response_rep, response_self_rep, response_cross_rep, response_inter_feat_multi], axis=-1)
            context_concat_dense_rep=self.dense1[k](context_concat_rep)
            context_concat_dense_rep=tf.nn.dropout(context_concat_dense_rep,self.config.dropout)
            response_concat_dense_rep=self.dense2[k](response_concat_rep)
            response_concat_dense_rep=tf.nn.dropout(response_concat_dense_rep,self.config.dropout)
            inter_feat = tf.matmul(context_rep, tf.transpose(response_rep, perm=[0, 2, 1])) / tf.sqrt(tf.cast(self.enc_units,dtype=tf.float32))
            inter_feat_self = tf.matmul(context_self_rep, tf.transpose(response_self_rep, perm=[0, 2, 1])) / tf.sqrt(tf.cast(self.enc_units,dtype=tf.float32))
            inter_feat_cross = tf.matmul(context_cross_rep, tf.transpose(response_cross_rep, perm=[0, 2, 1])) / tf.sqrt(tf.cast(self.enc_units,dtype=tf.float32))
            inter_feat_collection.append(inter_feat)
            inter_feat_collection.append(inter_feat_self)
            inter_feat_collection.append(inter_feat_cross)
            if k==0:
                context_rep = tf.add(context_rep, context_concat_dense_rep)
                response_rep = tf.add(response_rep, response_concat_dense_rep)
            else:
                context_rep = tf.add_n([parall_context_embeddings, context_rep, context_concat_dense_rep])
                response_rep = tf.add_n([expand_response_embeddings, response_rep, response_concat_dense_rep])
            context_rep=self.layer_morm1(context_rep)
            response_rep=self.layer_morm2(response_rep)
            context_rep = tf.multiply(context_rep, tf.expand_dims(parall_context_mask, axis=-1))
            response_rep = tf.multiply(response_rep, tf.expand_dims(expand_response_mask, axis=-1))
            matching_feat = tf.stack(inter_feat_collection, axis=-1)
            pool1_result=self.pool1[k](self.conv1[k](matching_feat))
            pool2_result=self.pool2[k](self.conv2[k](pool1_result))
            flatten_result=tf.nn.dropout(self.flatten(pool2_result),rate=self.config.dropout)
            matching_vector=self.dense3[k](flatten_result)
            matching_vector = tf.reshape(matching_vector, [-1, self.config.max_turn, self.config.hidden_size])
            _,last_hidden=self.encoder[k](matching_vector)
            logits=self.dense4[k](last_hidden)
            y_pred = tf.nn.softmax(logits)
            y_pred_list.append(y_pred)
            logits_list.append(logits)
        return logits_list,y_pred_list
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
    