from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import warnings
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.functional import dropout, linear, relu, softmax
from torch.nn.init import normal_
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.transformer import _get_activation_fn, TransformerDecoderLayer, TransformerDecoder, TransformerEncoder
from torch.nn.parameter import Parameter
from .att_model import pack_wrapper, AttModel


class WordEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(WordEmbeddings, self).__init__()
        self.d_model = d_model
        model_name_or_path = os.path.join(f"/opt/data/ARGON/containers/models/w2v/glove")

        if os.path.exists(model_name_or_path):
            embed_weight = np.load(os.path.join(model_name_or_path, 'w2v.npy'))
        else:
            print(" you should specify the w2v path")
        self.lut = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), padding_idx=0, freeze=False)

    def forward(self, x):
        embedded_captions = self.lut(x)
        self.lut(x) * math.sqrt(self.d_model)
        return embedded_captions

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class PositionalEncoding2D(PositionalEncoding):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding2D, self).__init__(d_model, max_len=max_len)

    def forward(self, x):
        pe_w = self.pe[:x.size(0), :].repeat(self.pe.shape[0], 1, 1)
        pe_h = self.pe[:x.size(0), :].repeat_interleave(self.pe.shape[0], dim=0)
        x = x + pe_w + pe_h
        return x


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 128*16*512, 11*16, None, 11*11
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), hidden_states, tgt_mask, src_mask)



class DecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, nlayer_enc, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        for i in range(nlayer_enc):
            setattr(self, 'linear_alpha{0}'.format(i), Linear(d_model * 2, d_model))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        mesh = None
        memory=memory.permute(1,0,2,3)
        ## new update!
        if memory.shape[2]!=tgt.shape[1]: memory=memory.reshape(memory.shape[0],98,-1,memory.shape[3])
        for i in range(memory.shape[0]):
            if len(memory.shape) == 5:
                tgt_stack = []
                for j in range(memory.shape[1]):
                    tgt2 = self.multihead_attn(tgt, memory[i, j], memory[i, j], attn_mask=memory_mask,
                                               key_padding_mask=memory_key_padding_mask)[0]
                    tgt_stack.append(tgt2)
                tgt_stack = torch.stack(tgt_stack, dim=0)
                tgt2, _ = torch.max(tgt_stack, dim=0)
            else:
                tgt2 = self.multihead_attn(tgt, memory[i], memory[i], attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)[0]
            mtgt = tgt + self.dropout2(tgt2)
            mtgt = self.norm2(mtgt)
            linear_alpha = getattr(self, 'linear_alpha{0}'.format(i))
            alpha = sigmoid(linear_alpha(torch.cat([tgt, mtgt], dim=2)))
            if mesh is None:
                mesh = mtgt * alpha
            else:
                mesh += mtgt * alpha
        mesh *= memory.shape[0] ** -0.5
        tgt = mesh

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Encoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__(encoder_layer, num_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        outputs = []
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        return output

class EncoderLayer(Module):
    def __init__(self, d_model, nhead, nmem, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionWithMem(d_model, nhead, nmem, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiheadAttentionWithMem(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, num_memory, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None):
        super(MultiheadAttentionWithMem, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                                                        kdim, vdim)
        self.num_memory = num_memory
        self.mem_k = Parameter(torch.Tensor(num_memory, 1, embed_dim))
        self.mem_v = Parameter(torch.Tensor(num_memory, 1, embed_dim))
        self._reset_mem_parameters()

    def _reset_mem_parameters(self):
        normal_(self.mem_k, 0.0, 1.0 / (self.embed_dim // self.num_heads))
        normal_(self.mem_v, 0.0, 1.0 / self.num_memory)

    @staticmethod
    def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, num_memory, in_proj_weight,
                                     in_proj_bias, bias_k, bias_v, mem_k, mem_v, add_zero_attn, dropout_p,
                                     out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
                                     need_weights=True, attn_mask=None, use_separate_proj_weight=False,
                                     q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None,
                                     static_v=None):
        qkv_same = torch.equal(query, key) and torch.equal(key, value)
        kv_same = torch.equal(key, value)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if use_separate_proj_weight is not True:
            if qkv_same:
                # self-attention
                q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

            elif kv_same:
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = linear(key, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = linear(value, _w, _b)
        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            if in_proj_bias is not None:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)
        q = q * scaling

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.cat([attn_mask,
                                           torch.zeros((attn_mask.size(0), 1),
                                                       dtype=attn_mask.dtype,
                                                       device=attn_mask.device)], dim=1)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                       dtype=key_padding_mask.dtype,
                                                       device=key_padding_mask.device)], dim=1)
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None

        # Additional memory process
        scaled_mem_k = ((embed_dim // num_heads) ** 0.5) * mem_k.expand(num_memory, k.shape[1], k.shape[2])
        scaled_mem_v = (num_memory ** 0.5) * mem_v.expand(num_memory, v.shape[1], v.shape[2])
        k = torch.cat([k, scaled_mem_k], dim=0)
        v = torch.cat([v, scaled_mem_v], dim=0)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                              dtype=attn_mask.dtype,
                                                              device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.num_memory,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.mem_k, self.mem_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.num_memory,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.mem_k, self.mem_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

class EncoderDecoder(AttModel):

    def make_model(self, vocab_size):
        c = copy.deepcopy
        position = PositionalEncoding(self.d_model)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, nhead=self.num_heads, nmem=self.nmem), self.encoder_num_layers),
            TransformerDecoder(
                DecoderLayer(self.d_model, nhead=self.num_heads, nlayer_enc=self.encoder_num_layers),
                self.decoder_num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, vocab_size), c(position)),
            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.encoder_num_layers = args.encoder_num_layers
        self.decoder_num_layers = args.decoder_num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.nmem=args.m2_memory
        self.dropout = args.dropout

        vocab_size = self.vocab_size + 1

        self.model = self.make_model(vocab_size)
        self.logit = nn.Linear(args.d_model, vocab_size)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        att_feats = att_feats.transpose(0, 1)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        #att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            #seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
            seq_mask = self.generate_square_subsequent_mask(seq.size(-1)).to(seq.device)
            seq = seq.transpose(0, 1)
        else:
            seq_mask = None

        return att_feats, seq, None, seq_mask

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs.permute(1,0,2)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out = self.model.decode(memory, mask, ys.permute(1,0), tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(memory.device))
        out = out.transpose(0, 1)
        return out[:, -1], [ys.unsqueeze(0)]
