import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant
import warnings
from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from typing import List, Optional, Tuple, Union
import math
import types
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralMLP,
    apply_rotary_pos_emb,
    repeat_kv
)
from transformers.cache_utils import Cache
from utils.utils import parse_args,load_llava

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = 2048
    return model


class Delta(nn.Module):
    def __init__(self,base,U,S,V):
        super().__init__()
        self.register_buffer("base", base)
        self.register_buffer("U", None)
        self.register_buffer("V", None)
        self.register_buffer("S", None)
        
        self.register_buffer("U_total", U)
        self.register_buffer("V_total", V)
        self.register_buffer("S_total", S)

    def pre_quant(self,cur_col,pre_col=0):
        self.U = self.U_total[:,pre_col:cur_col] 
        self.S = self.S_total[pre_col:cur_col]
        self.V = self.V_total[:,pre_col:cur_col]
    
    def post_quant(self,bit,name):
        if args.save_trained_path is not None:
            tmp[name + f".U_{bit}"] = self.U
            tmp[name + f".S_{bit}"] = self.S
            tmp[name + f".V_{bit}"] = self.V
            
            if tmp.get(name + ".base") is None:
                tmp[name + ".base"] = self.base

        self.base = self.base + self.U @ torch.diag(self.S) @ self.V.T


    
    def forward(self, x, gptq=None,quant_type=None):
        # TODO: This can be faster
        
        if gptq is not None:
            if quant_type == "V":
                gptq.add_batch(x[0].data, x[0].data)
            else:
                y = x.clone()
                w_ = (torch.diag(self.S) @ self.V.T).to(x.dtype)
                y = y @ w_.T
                
                gptq.add_batch(y[0].data, y[0].data)
                
    
        w = (self.base + self.U @ torch.diag(self.S) @ self.V.T).to(x.dtype)
        return x @ w.T


def mistral_quant_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    quant_type = gptqs.get("quant_type",None)
    gptq_q , gptq_k , gptq_v , gptq_o = None, None, None, None
    
    if len(gptqs.keys()) == 4:
        for k,v in gptqs.items():
            if "q_proj" in k:
                gptq_q = v
            elif "k_proj" in k:
                gptq_k = v
            elif "v_proj" in k:
                gptq_v = v
    else:
        for k,v in gptqs.items():
            if "o_proj" in k:
                gptq_o = v    
    
    query_states = self.q_proj(hidden_states,gptq_q,quant_type)
    key_states = self.k_proj(hidden_states,gptq_k,quant_type)
    value_states = self.v_proj(hidden_states,gptq_v,quant_type)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output,gptq_o,quant_type)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def mistral_quant_mlp_forward(self, x):
    
    quant_type = gptqs.get("quant_type",None)
    gptq_up , gptq_gate , gptq_down  = None, None, None
    
    if len(gptqs.keys()) == 3:
        for k,v in gptqs.items():
            if "up_proj" in k:
                gptq_up = v
            elif "gate_proj" in k:
                gptq_gate = v
    else:
        for k,v in gptqs.items():
            if "down_proj" in k:
                gptq_down = v     
    
    return self.down_proj(self.act_fn(self.gate_proj(x,gptq_gate,quant_type)) * self.up_proj(x,gptq_up,quant_type),gptq_down,quant_type)


def enable_mistral_quant_forward(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_mistral_quant_forward(
                module,
            )
        
        if isinstance(module, MistralAttention):
            model._modules[name].forward = types.MethodType(
                mistral_quant_attn_forward, model._modules[name]
            )
            
        if isinstance(module, MistralMLP): 
            model._modules[name].forward = types.MethodType(
                mistral_quant_mlp_forward, model._modules[name]
            )
        
gptqs = {} 
@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    
    state_dict = torch.load(args.saved_delta_path)
    
    for name, module in model.named_modules():
        if "vision" in name:
            continue        
        
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    setattr(module, subname, None)
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    base = state_dict[name + "." + subname + ".base"]
                    U = state_dict[name + "." + subname + ".U"]
                    V = state_dict[name + "." + subname + ".V"]
                    S = state_dict[name + "." + subname + ".S"]
                    
                    setattr(module, subname, Delta(base=base, U=U, S=S, V=V))
    
    
    enable_mistral_quant_forward(model)
    
    bits = args.bits

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        if args.attn_fp16_col != 0 or args.mlp_fp16_col != 0:
            params = ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj','self_attn.o_proj','mlp.up_proj', 'mlp.gate_proj','mlp.down_proj']
            for param in params:
                cur_col = args.attn_fp16_col if "self_attn" in param else args.mlp_fp16_col
                layer.get_submodule(param).pre_quant(pre_col=0,cur_col=cur_col)
                layer.get_submodule(param).post_quant(bit=16,name=f"model.layers.{i}." + param) 
         
        for bit in bits:
            print(f'Quantizing  layer {bit}bit {i+1}/{len(layers)}..')
            print('+------------------+--------------+------------+-----------+-------+')
            print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
            print('+==================+==============+============+===========+=======+')
            if args.true_sequential:
                sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],['self_attn.o_proj'],['mlp.up_proj', 'mlp.gate_proj'],['mlp.down_proj']]

            for names in sequential: 
                for name in names:
                    if bit == 8:
                        if "self_attn" in name:
                            pre_col, cur_col = args.attn_fp16_col, args.attn_fp16_col + args.attn_int8_col
                        else:
                            pre_col, cur_col = args.mlp_fp16_col , args.mlp_fp16_col + args.mlp_int8_col
                    elif bit == 4:
                        if "self_attn" in name:
                            pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col, args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col  
                        else: 
                            pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col, args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col  
                    elif bit == 3:
                        if "self_attn" in name:
                            pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col ,args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col  
                        else: 
                            pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col , args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col  
                    elif bit == 2:
                        if "self_attn" in name:
                            pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col ,args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col + args.attn_int2_col  
                        else: 
                            pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col , args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col + args.mlp_int2_col                    
                        
                    layer.get_submodule(name).pre_quant(pre_col=pre_col,cur_col=cur_col)
                            
            for names in sequential:
                for ii in range(2):
                    quant_type = "V" if ii == 0 else "U"
                    
                    for name in names:
                        name = f"model.layers.{i}." + name        
                        gptq = GPTQ(model.get_submodule(name), quant_type=quant_type,observe=args.observe)
                        gptq.quantizer.configure(bit, perchannel=True, sym=args.sym, mse=False)
                        gptqs[name] = gptq
                    
                    gptqs["quant_type"] = quant_type
                    for j in range(args.nsamples):
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

                    for k,v in gptqs.items():
                        if "proj" in k:
                            gptq = v
                            scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=k.rsplit(".")[-1] + f".{quant_type}")
                    
                    gptqs.clear()

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            
            for names in sequential:
                for name in names:
                    if bit != bits[-1]:
                        layer.get_submodule(name).post_quant(bit=bit,name=f"model.layers.{i}." + name) 
            
            inps, outs = outs, inps
            print('+------------------+--------------+------------+-----------+-------+')
            print('\n')
            
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

    '''
    with torch.no_grad():
        for layer in layers:
            for name, module in layer.named_modules():
                if "proj" in name:
                    
                    if "self_attn" in name:
                        pre_col, cur_col = args.int8_col + args.attn_int4_col, args.int8_col + args.attn_int4_col + args.attn_int1_col  
                    else:
                        pre_col, cur_col = args.int8_col + args.mlp_int4_col, args.int8_col + args.mlp_int4_col + args.mlp_int1_col  
                    
                    layer.get_submodule(name).pre_quant(pre_col=pre_col,cur_col=cur_col)
                    U,V = layer.get_submodule(name).U, layer.get_submodule(name).V
                    U_total,V_total = layer.get_submodule(name).U_total, layer.get_submodule(name).V_total
                    coeff_u, coeff_v = U_total.float().abs().mean() , V_total.float().abs().mean()
                    sign_u,sign_v = torch.sign(U) , torch.sign(V)
                    mask_u , mask_v = sign_u == 0 , sign_v == 0 
                    sign_u[mask_u] = 1 
                    sign_v[mask_v] = 1
                    U , V = sign_u * coeff_u, sign_v * coeff_v
                    # coeff_test = delta_test.float().abs().mean()
                    # signs = torch.sign(delta_test)
                    # mask = signs == 0 
                    # signs[mask] = 1
                    # delta_test = signs * coeff_test
    '''
    
    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...') 

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)

@torch.no_grad()
def save_compressed_delta(save_compressed_delta_dir,model):
    compressed_delta = dict()
    
    for name, module in model.named_modules():
        
        if "vision_tower" in name:
            continue
        
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    base = model.get_submodule(name + "." + subname).base
                    
                    U,S,V = model.get_submodule(name + "." + subname).U, model.get_submodule(name + "." + subname).S , model.get_submodule(name + "." + subname).V
                    delta = (U @ torch.diag(S) @ V.t())
                    
                    if args.save_trained_path is not None:
                        tmp[name + "." + subname + f".U_{args.bits[-1]}"] = U
                        tmp[name + "." + subname + f".S_{args.bits[-1]}"] = S
                        tmp[name + "." + subname + f".V_{args.bits[-1]}"] = V
                    compressed_delta[name + "." + subname + ".weight"] = (base + delta).to(torch.bfloat16)
    
    torch.save(compressed_delta, save_compressed_delta_dir)
    if args.save_trained_path is not None:
        torch.save(tmp, args.save_trained_path)

if __name__ == '__main__':
    args = parse_args()
    
    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        if "llava" not in args.model.lower():
            model = get_llama(args.model)
            model.eval()
        else:
            model = load_llava(args.model,"cuda" if torch.cuda.is_available() else "cpu")
            if not hasattr(model, 'seqlen'):
                model.seqlen = 2048

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)
    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        
        tmp = dict()
        
        quantizers = llama_sequential(model, dataloader, DEV)

        if args.save_compressed_delta_dir is not None:
            save_compressed_delta(args.save_compressed_delta_dir,model)
        
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            llama_eval(model, testloader, DEV)
    
    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)
        
    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)
