import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_delta(model_name_or_path, delta_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.bfloat16).to(device)
    delta = torch.load(delta_path)
    
    model.load_state_dict(delta,strict=False)
    return tokenizer, model

def decomposition(masked_input_tensor,dim=None):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    
    outlier_U , outlier_V = None, None
    
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    
    return U, S, V 


def load_model(base_model,finetuned_model,dim_attn,save_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model,torch_dtype=torch.bfloat16).to(device)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model,torch_dtype=torch.bfloat16).to(device)
    
    param_dict = dict()
    for k,v in base_model.state_dict().items():
        if "self_attn" in k or "mlp" in k:
            if ".weight" in k:
                delta = finetuned_model.state_dict()[k] - v
                dim = dim_attn
                
                if "mlp" in k:
                    dim = int(dim * 1.45)
                U,S,V = decomposition(delta, dim=dim)
                
                k = k.replace(".weight", "")
                
                param_dict[k + ".base"] = v
                param_dict[k + ".U"] = U.data.to(torch.bfloat16)
                param_dict[k + ".S"] = S.data.to(torch.bfloat16)
                param_dict[k + ".V"] = V.data.to(torch.bfloat16)
            
                
            
    torch.save(param_dict, save_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_svd', action='store_true', help='llama model to load')
    parser.add_argument('--merge', action='store_true', help='llama model to load')
    parser.add_argument('--dim', type=int, default=256, help='llama model to load')
    parser.add_argument('--delta_path', type=str, default="", help='llama model to load')
    parser.add_argument('--save_path', type=str, default="", help='llama model to load')
    parser.add_argument('--fintuned_model', type=str, default="", help='llama model to load')    
    args = parser.parse_args()
    
    if args.use_svd:
        base_model = "/path/to/base/model"
        finetuned_model = args.fintuned_model
        dim = args.dim
        save_path = f"/path/to/save/model"
        
        load_model(base_model=base_model,finetuned_model=finetuned_model,dim_attn=dim,save_path=save_path)
    elif args.merge:
        model_name_or_path = args.fintuned_model
        delta_path = args.delta_path
        save_path = args.save_path
        
        tokenizer , model = load_delta(model_name_or_path=model_name_or_path, delta_path=delta_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)