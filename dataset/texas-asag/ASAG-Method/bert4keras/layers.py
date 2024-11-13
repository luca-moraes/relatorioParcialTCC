
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, is_tf_keras
from bert4keras.backend import sequence_masking
from bert4keras.backend import recompute_grad
from keras import initializers, activations
from keras.layers import *


def integerize_shape(func):
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func


if (not is_tf_keras) and keras.__version__ < '2.3':

    class Layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True

        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                if not hasattr(self, '_layers'):
                    self._layers = []
                if value not in self._layers:
                    self._layers.append(value)
            super(Layer, self).__setattr__(name, value)

        @property
        def trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            if trainable:
                trainable_weights = super(Layer, self).trainable_weights[:]
                for l in getattr(self, '_layers', []):
                    trainable_weights += l.trainable_weights
                return trainable_weights
            else:
                return []

        @property
        def non_trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            non_trainable_weights = super(Layer, self).non_trainable_weights[:]
            for l in getattr(self, '_layers', []):
                if trainable:
                    non_trainable_weights += l.non_trainable_weights
                else:
                    non_trainable_weights += l.weights
            return non_trainable_weights


    if keras.__version__ < '2.2.5':

        import inspect


        class Model(keras.models.Model):
            def fit(self, x=None, *args, **kwargs):
                if inspect.isgenerator(x):
                    return self.fit_generator(x, *args, **kwargs)
                else:
                    return super(Model, self).fit(x, *args, **kwargs)


        keras.models.Model = Model

else:

    class Layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True

if (not is_tf_keras) or tf.__version__ < '1.15':

    if not is_tf_keras:
        NodeBase = keras.engine.base_layer.Node
    else:
        from tensorflow.python.keras.engine import base_layer

        NodeBase = base_layer.Node


    class Node(NodeBase):
        @property
        def arguments(self):
            return self._arguments.copy()

        @arguments.setter
        def arguments(self, value):
            self._arguments = value or {}


    if not is_tf_keras:
        keras.engine.base_layer.Node = Node
    else:
        base_layer.Node = Node


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    def call(self, inputs, mask=None):
        axis = 1 if self.data_format == 'channels_last' else 2
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = mask[..., None] if axis == 1 else mask[:, None]
            return K.sum(inputs * mask, axis=axis) / K.sum(mask, axis=axis)
        else:
            return K.mean(inputs, axis=axis)


class GlobalMaxPooling1D(keras.layers.GlobalMaxPooling1D):
    def __init__(self, data_format='channels_last', **kwargs):
        super(GlobalMaxPooling1D, self).__init__(data_format, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        axis = 1 if self.data_format == 'channels_last' else 2
        inputs = sequence_masking(inputs, mask, '-inf', axis)
        return K.max(inputs, axis=axis)

    def compute_mask(self, inputs, mask=None):
        return None


keras.layers.GlobalAveragePooling1D = GlobalAveragePooling1D
keras.layers.GlobalMaxPooling1D = GlobalMaxPooling1D


class Embedding(keras.layers.Embedding):
    def compute_mask(self, inputs, mask=None):
        if K.ndim(inputs) == 2:
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return K.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = K.transpose(self.embeddings)
            return K.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (K.int_shape(self.embeddings)[0],)


class BiasAdd(Layer):
    @integerize_shape
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias', shape=(output_dim,), initializer='zeros'
        )

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)


class Scale(Layer):
    def __init__(self, scale=1, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = {
            'scale': self.scale,
        }
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Concatenate1D(Layer):
    def call(self, inputs):
        return K.concatenate(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            masks = []
            for i, m in enumerate(mask):
                if m is None:
                    m = K.ones_like(inputs[i][..., 0], dtype='bool')
                masks.append(m)
            return K.concatenate(masks, axis=1)

    def compute_output_shape(self, input_shape):
        if all([shape[1] for shape in input_shape]):
            seq_len = sum([shape[1] for shape in input_shape])
            return (input_shape[0][0], seq_len, input_shape[0][2])
        else:
            return (input_shape[0][0], None, input_shape[0][2])


class BatchSplit(Layer):
    def __init__(self, parts, **kwargs):
        super(BatchSplit, self).__init__(**kwargs)
        self.parts = parts

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return [o for i in mask for o in self.compute_mask(inputs, i)]

        if mask is not None:
            return self.call(mask)
        elif np.ndim(self.parts) > 0:
            return [None] * len(self.parts)
        else:
            return [None] * self.parts

    def call(self, inputs):
        if isinstance(inputs, list):
            return [o for i in inputs for o in self.call(i)]

        outputs = []

        batch_size = K.shape(inputs)[0]
        if np.ndim(self.parts) > 0:
            batch_size = K.cast(batch_size, 'float64')
            slices = [
                K.cast(p * batch_size / sum(self.parts), 'int32')
                for p in np.cumsum(self.parts).astype('float64')
            ]
        else:
            stride = K.cast(
                tf.math.ceil(batch_size / self.parts), K.dtype(batch_size)
            )
            slices = [stride * (i + 1) for i in range(self.parts)]

        for i, _ in enumerate(slices):
            if i == 0:
                outputs.append(inputs[:slices[0]])
            elif i == len(slices) - 1:
                outputs.append(inputs[slices[-2]:])
            else:
                outputs.append(inputs[slices[i - 1]:slices[i]])

        return outputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return [
                o for i in input_shape for o in self.compute_output_shape(i)
            ]

        if np.ndim(self.parts) > 0:
            return [input_shape] * len(self.parts)
        else:
            return [input_shape] * self.parts

    def get_config(self):
        config = {
            'parts': self.parts,
        }
        base_config = super(BatchSplit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchConcat(Layer):
    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            if all([m is not None for m in mask]):
                return K.concatenate(mask, 0)

    def call(self, inputs):
        return K.concatenate(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MultiHeadAttention(Layer):

    def __init__(
            self,
            heads,
            head_size,
            out_dim=None,
            key_size=None,
            use_bias=True,
            attention_scale=True,
            attention_dropout=None,
            return_attention_scores=False,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]

        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)

        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)

        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        o = self.o_dense(o)

        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):

        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            a_bias = inputs[n]
            n += 1
        if p_bias == 'rotary':
            cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)

        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif p_bias == 't5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            a = a + K.expand_dims(position_bias, 0)

        if self.attention_scale:
            a = a / self.key_size ** 0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = K.softmax(a)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)

        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            conditional=False,
            hidden_units=None,
            hidden_activation='linear',
            hidden_initializer='glorot_uniform',
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    @recompute_grad
    def call(self, inputs):
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    def __init__(
            self,
            input_dim,
            output_dim,
            merge_mode='add',
            hierarchical=None,
            embeddings_initializer='zeros',
            custom_position_ids=False,
            **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SinusoidalPositionEmbedding(Layer):
    def __init__(
            self,
            output_dim,
            merge_mode='add',
            custom_position_ids=False,
            **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RelativePositionEmbedding(Layer):
    def __init__(
            self, input_dim, output_dim, embeddings_initializer='zeros', **kwargs
    ):
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs):
        pos_ids = self.compute_position_ids(inputs)
        return K.gather(self.embeddings, pos_ids)

    def compute_position_ids(self, inputs):
        q, v = inputs
        q_idxs = K.arange(0, K.shape(q)[1], dtype='int32')
        q_idxs = K.expand_dims(q_idxs, 1)
        v_idxs = K.arange(0, K.shape(v)[1], dtype='int32')
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        max_position = (self.input_dim - 1) // 2
        pos_ids = K.clip(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return pos_ids

    def compute_output_shape(self, input_shape):
        return (None, None, self.output_dim)

    def compute_mask(self, inputs, mask):
        return mask[0]

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RelativePositionEmbeddingT5(RelativePositionEmbedding):
    def __init__(
            self,
            input_dim,
            output_dim,
            max_distance=128,
            bidirectional=True,
            embeddings_initializer='zeros',
            **kwargs
    ):
        super(RelativePositionEmbeddingT5,
              self).__init__(input_dim, output_dim, **kwargs)
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def compute_position_ids(self, inputs):
        q, v = inputs
        q_idxs = K.arange(0, K.shape(q)[1], dtype='int32')
        q_idxs = K.expand_dims(q_idxs, 1)
        v_idxs = K.arange(0, K.shape(v)[1], dtype='int32')
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        num_buckets, max_distance = self.input_dim, self.max_distance
        ret = 0
        n = -pos_ids
        if self.bidirectional:
            num_buckets //= 2
            ret += K.cast(K.less(n, 0), 'int32') * num_buckets
            n = K.abs(n)
        else:
            n = K.maximum(n, 0)
        
        max_exact = num_buckets // 2
        is_small = K.less(n, max_exact)
        val_if_large = max_exact + K.cast(
            K.log(K.cast(n, K.floatx()) / max_exact) /
            np.log(max_distance / max_exact) * (num_buckets - max_exact),
            'int32',
        )
        val_if_large = K.minimum(val_if_large, num_buckets - 1)
        ret += K.switch(is_small, n, val_if_large)
        return ret

    def get_config(self):
        config = {
            'max_distance': self.max_distance,
            'bidirectional': self.bidirectional,
        }
        base_config = super(RelativePositionEmbeddingT5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    def __init__(
            self,
            units,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        for i, activation in enumerate(self.activation):
            i_dense = Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalRandomField(Layer):
    def __init__(self, lr_multiplier=1, **kwargs):
        super(ConditionalRandomField, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier

    @integerize_shape
    def build(self, input_shape):
        super(ConditionalRandomField, self).build(input_shape)
        output_dim = input_shape[-1]
        self._trans = self.add_weight(
            name='trans',
            shape=(output_dim, output_dim),
            initializer='glorot_uniform'
        )
        if self.lr_multiplier != 1:
            K.set_value(self._trans, K.eval(self._trans) / self.lr_multiplier)

    @property
    def trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._trans
        else:
            return self._trans

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        return sequence_masking(inputs, mask, '-inf', 1)

    def target_score(self, y_true, y_pred):
        point_score = tf.einsum('bni,bni->b', y_true, y_pred)  
        trans_score = tf.einsum(
            'bni,ij,bnj->b', y_true[:, :-1], self.trans, y_true[:, 1:]
        )  
        return point_score + trans_score

    def log_norm_step(self, inputs, states):
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  
        trans = K.expand_dims(self.trans, 0)  
        outputs = tf.reduce_logsumexp(
            states + trans, 1
        )  
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def dense_loss(self, y_true, y_pred):
        
        mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
        mask = K.cast(mask, K.floatx())
        
        y_true, y_pred = y_true * mask, y_pred * mask
        target_score = self.target_score(y_true, y_pred)
        
        init_states = [y_pred[:, 0]]
        y_pred = K.concatenate([y_pred, mask], axis=2)
        input_length = K.int_shape(y_pred[:, 1:])[1]
        log_norm, _, _ = K.rnn(
            self.log_norm_step,
            y_pred[:, 1:],
            init_states,
            input_length=input_length
        )  
        log_norm = tf.reduce_logsumexp(log_norm, 1)  
        
        return log_norm - target_score

    def sparse_loss(self, y_true, y_pred):
        
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        
        y_true = K.one_hot(y_true, K.shape(self.trans)[0])
        return self.dense_loss(y_true, y_pred)

    def dense_accuracy(self, y_true, y_pred):
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def sparse_accuracy(self, y_true, y_pred):
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(isequal * mask) / K.sum(mask)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
        }
        base_config = super(ConditionalRandomField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaximumEntropyMarkovModel(Layer):

    def __init__(self, lr_multiplier=1, hidden_dim=None, **kwargs):
        super(MaximumEntropyMarkovModel, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  
        self.hidden_dim = hidden_dim  

    @integerize_shape
    def build(self, input_shape):
        super(MaximumEntropyMarkovModel, self).build(input_shape)
        output_dim = input_shape[-1]

        if self.hidden_dim is None:
            self._trans = self.add_weight(
                name='trans',
                shape=(output_dim, output_dim),
                initializer='glorot_uniform'
            )
            if self.lr_multiplier != 1:
                K.set_value(
                    self._trans,
                    K.eval(self._trans) / self.lr_multiplier
                )
        else:
            self._l_trans = self.add_weight(
                name='l_trans',
                shape=(output_dim, self.hidden_dim),
                initializer='glorot_uniform'
            )
            self._r_trans = self.add_weight(
                name='r_trans',
                shape=(output_dim, self.hidden_dim),
                initializer='glorot_uniform'
            )

            if self.lr_multiplier != 1:
                K.set_value(
                    self._l_trans,
                    K.eval(self._l_trans) / self.lr_multiplier
                )
                K.set_value(
                    self._r_trans,
                    K.eval(self._r_trans) / self.lr_multiplier
                )

    @property
    def trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._trans
        else:
            return self._trans

    @property
    def l_trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._l_trans
        else:
            return self._l_trans

    @property
    def r_trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._r_trans
        else:
            return self._r_trans

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        return sequence_masking(inputs, mask, '-inf', 1)

    def reverse_sequence(self, inputs, mask=None):
        if mask is None:
            return [x[:, ::-1] for x in inputs]
        else:
            length = K.cast(K.sum(mask, 1), 'int32')
            return [tf.reverse_sequence(x, length, seq_axis=1) for x in inputs]

    def basic_loss(self, y_true, y_pred, go_backwards=False):
        
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        
        if self.hidden_dim is None:
            if go_backwards:  
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                trans = K.transpose(self.trans)
            else:
                trans = self.trans
            histoty = K.gather(trans, y_true)
        else:
            if go_backwards:  
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                r_trans, l_trans = self.l_trans, self.r_trans
            else:
                l_trans, r_trans = self.l_trans, self.r_trans
            histoty = K.gather(l_trans, y_true)
            histoty = tf.einsum('bnd,kd->bnk', histoty, r_trans)
        
        histoty = K.concatenate([y_pred[:, :1], histoty[:, :-1]], 1)
        y_pred = (y_pred + histoty) / 2
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        return K.sum(loss * mask) / K.sum(mask)

    def sparse_loss(self, y_true, y_pred):
        loss = self.basic_loss(y_true, y_pred, False)
        loss = loss + self.basic_loss(y_true, y_pred, True)
        return loss / 2

    def dense_loss(self, y_true, y_pred):
        y_true = K.argmax(y_true, 2)
        return self.sparse_loss(y_true, y_pred)

    def basic_accuracy(self, y_true, y_pred, go_backwards=False):
        
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        
        if self.hidden_dim is None:
            if go_backwards:  
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                trans = K.transpose(self.trans)
            else:
                trans = self.trans
            histoty = K.gather(trans, y_true)
        else:
            if go_backwards:  
                y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
                r_trans, l_trans = self.l_trans, self.r_trans
            else:
                l_trans, r_trans = self.l_trans, self.r_trans
            histoty = K.gather(l_trans, y_true)
            histoty = tf.einsum('bnd,kd->bnk', histoty, r_trans)
        
        histoty = K.concatenate([y_pred[:, :1], histoty[:, :-1]], 1)
        y_pred = (y_pred + histoty) / 2
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(isequal * mask) / K.sum(mask)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        accuracy = self.basic_accuracy(y_true, y_pred, False)
        accuracy = accuracy + self.basic_accuracy(y_true, y_pred, True)
        return accuracy / 2

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
            'hidden_dim': self.hidden_dim,
        }
        base_config = super(MaximumEntropyMarkovModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalPointer(Layer):
    def __init__(
            self,
            heads,
            head_size,
            RoPE=True,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(GlobalPointer, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(GlobalPointer, self).build(input_shape)
        self.dense = Dense(
            units=self.head_size * self.heads * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def compute_mask(self, inputs, mask=None):
        return None

    @recompute_grad
    def call(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = tf.split(inputs, self.heads, axis=-1)
        inputs = K.stack(inputs, axis=-2)
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = K.repeat_elements(pos[..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(pos[..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        logits = tf.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        mask = tf.linalg.band_part(K.ones_like(logits), 0, -1)
        logits = logits - (1 - mask) * K.infinity()
        return logits / self.head_size ** 0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.heads, input_shape[1], input_shape[1])

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'RoPE': self.RoPE,
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GlobalPointer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Loss(Layer):

    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss, inputs=inputs)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, list):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def compute_mask(self, inputs, mask):
        if mask is not None:
            if self.output_axis is None:
                return mask
            elif isinstance(self.output_axis, list):
                return [mask[i] for i in self.output_axis]
            else:
                return mask[self.output_axis]

    def get_config(self):
        config = {
            'output_axis': self.output_axis,
        }
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'Embedding': Embedding,
    'BiasAdd': BiasAdd,
    'Scale': Scale,
    'Concatenate1D': Concatenate1D,
    'BatchSplit': BatchSplit,
    'BatchConcat': BatchConcat,
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'PositionEmbedding': PositionEmbedding,
    'SinusoidalPositionEmbedding': SinusoidalPositionEmbedding,
    'RelativePositionEmbedding': RelativePositionEmbedding,
    'RelativePositionEmbeddingT5': RelativePositionEmbeddingT5,
    'FeedForward': FeedForward,
    'ConditionalRandomField': ConditionalRandomField,
    'MaximumEntropyMarkovModel': MaximumEntropyMarkovModel,
    'GlobalPointer': GlobalPointer,
    'Loss': Loss,
}

keras.utils.get_custom_objects().update(custom_objects)