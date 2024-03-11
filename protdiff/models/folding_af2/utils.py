import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional,List, Dict, Callable

class MultiArgsSequential(nn.Sequential):
    """
        a hack to nn.Sequential to permit multi args input. 
        should ensure input and output tensor number are all same.
        all args should be position arg, not key args
        This is not friendly to torchscript
    """
    def forward(self, *inputs):
        for module in self:
            inputs = module(*inputs)
        return inputs

def checkpoint_function(function, *args):
    if not torch.is_grad_enabled():
        return function(*args)
    else:
        return checkpoint(function, *args)


def checkpoint_sequential(functions, segments, input, **kwargs):
    r"""
        A hack to checkpoint_sequential in torch.utils.checkpoint,
        to support multi args
        input: args to sequential, as tuple
    """
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(*input):
            for j in range(start, end + 1):
                input = functions[j](*input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())
    
    def wrap_args(a):
        return (a,) if type(a) is not tuple else a
    if not torch.is_grad_enabled():
        return run_function(0, len(functions)-1, functions)(*wrap_args(input))
    

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
  
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(run_function(start, end, functions), *wrap_args(input),
                           preserve_rng_state=preserve)
    return checkpoint(run_function(end + 1, len(functions) - 1, functions),*wrap_args(input), preserve_rng_state=preserve)



# def checkpoint_sequential(functions, segments, input, **kwargs):
#     r"""
#         A hack to checkpoint_sequential in torch.utils.checkpoint,
#         to support multi args
#         input: args to sequential, as tuple
#     """
#     # Hack for keyword-only parameter in a python 2.7-compliant way
#     preserve = kwargs.pop('preserve_rng_state', True)
#     if kwargs:
#         raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

#     def run_function(start, end, functions):
#         def forward(*input):
#             for j in range(start, end + 1):
#                 input = functions[j](*input)
#             return input
#         return forward

#     if isinstance(functions, torch.nn.Sequential):
#         functions = list(functions.children())
    
#     def wrap_args(a):
#         return (a,) if type(a) is not tuple else a
#     if not torch.is_grad_enabled():
#         return run_function(0, len(functions)-1, functions)(*wrap_args(input))
    

#     segment_size = len(functions) // segments
#     # the last chunk has to be non-volatile
#     end = -1
  
#     for start in range(0, segment_size * (segments - 1), segment_size):
#         end = start + segment_size - 1
#         input = checkpoint(run_function(start, end, functions), *wrap_args(input),
#                            preserve_rng_state=preserve)
#     return checkpoint(run_function(end + 1, len(functions) - 1, functions),*wrap_args(input), preserve_rng_state=preserve)



# class ResModule(nn.Module):
#     def __init__(
#         self, module:nn.Module, dropout:Optional[nn.Module] = None, 
#         input_indices=(0,1,2,3), output_index=0,
#         name= None
#     ):
#         super().__init__()
#         self._name= name if name is not None else "ResModule_Unknown"
#         self.module= module
#         self.dropout= dropout
#         self.input_indices= input_indices
#         self.output_index= output_index
    
#     def __repr__(self) -> str:
#         return self._name
    
#     def forward(self, *args):
#         in_args= [args[i] for i in self.input_indices]
       
#         out = self.module(*in_args)
#         if self.dropout:
#             out= self.dropout(out)
        
#         outputs= list(args)
#         outputs[self.output_index] =  outputs[self.output_index] +out
#         return tuple(outputs)

# def sharded_layer(
#     layer:Callable,
#     args:List[torch.Tensor],
#     subbatch_size:int= 1024,
#     num_batch_dims:int =2,
# ):
#     """
#         layer: function or nn.Module, should return only one tensor
#         args: args for layer
#         subbatch_size: 
#         num_batch_dims:
#     """
#     bshapes= [a.shape[:num_batch_dims] for a in args]
#     # bshapes can only be max(bs) or 1
#     ex_bshapes= tuple([max(s) for s in zip(*bshapes)])
#     batch_size=1
#     for s in ex_bshapes:
#         batch_size*=s
#     if batch_size <= subbatch_size:
#         return layer(*args)
#     flat_args =[]
#     for arg in args:
#         # not all broadcast, then expand
#         if not sum(arg.shape[:num_batch_dims]) == num_batch_dims:
#             arg = arg.expand(*ex_bshapes, *arg.shape[num_batch_dims:])
#         arg= arg.reshape(-1, *arg.shape[num_batch_dims:])
#         flat_args.append(arg)
        
#     nchunks= (batch_size + subbatch_size -1)//subbatch_size
#     outs=[]
#     for i in range(nchunks):
#         start= i* subbatch_size
#         end= min((i+1)*subbatch_size, batch_size)
#         curr_batch = [t[start:end] if t.shape[0] >1 else t for t in flat_args]
#         chunk_out= checkpoint_function(layer, * curr_batch)
#         outs.append(chunk_out)
#     outs= torch.cat(outs, dim=0)
#     outs= outs.reshape(*ex_bshapes, *outs.shape[1:])
#     return outs


class ResModule(nn.Module):
    def __init__(
        self, module:nn.Module, dropout:Optional[nn.Module] = None, 
        input_indices=(0,1,2,3), output_index=0,
        name= None
    ):
        super().__init__()
        self._name= name if name is not None else "ResModule_Unknown"
        self.module= module
        self.dropout= dropout
        self.input_indices= input_indices
        self.output_index= output_index
    
    def __repr__(self) -> str:
        return self._name
    
    def forward(self, *args):
        in_args= [args[i] for i in self.input_indices]
       
        out = self.module(*in_args)
        if self.dropout:
            out= self.dropout(out)
        
        outputs= list(args)
        if isinstance(out, torch.Tensor):
            outputs[self.output_index] =  outputs[self.output_index] +out
        else:
            for order_idx, out_idx in enumerate(self.output_index):
                outputs[out_idx] =  outputs[out_idx] + out[order_idx]
        return tuple(outputs)

def sharded_layer(
    layer:Callable,
    args:List[torch.Tensor],
    subbatch_size:int= 1024,
    num_batch_dims:int =2,
):
    """
        layer: function or nn.Module, should return only one tensor
        args: args for layer
        subbatch_size: 
        num_batch_dims:
    """
    bshapes= [a.shape[:num_batch_dims] for a in args]
    # bshapes can only be max(bs) or 1
    ex_bshapes= tuple([max(s) for s in zip(*bshapes)])
    batch_size=1
    for s in ex_bshapes:
        batch_size*=s
    if batch_size <= subbatch_size:
        return layer(*args)
    flat_args =[]
    for arg in args:
        # not all broadcast, then expand
        if not sum(arg.shape[:num_batch_dims]) == num_batch_dims:
            arg = arg.expand(*ex_bshapes, *arg.shape[num_batch_dims:])
        arg= arg.reshape(-1, *arg.shape[num_batch_dims:])
        flat_args.append(arg)
        
    nchunks= (batch_size + subbatch_size -1)//subbatch_size
    outs=[]
    for i in range(nchunks):
        start= i* subbatch_size
        end= min((i+1)*subbatch_size, batch_size)
        curr_batch = [t[start:end] if t.shape[0] >1 else t for t in flat_args]
        chunk_out= checkpoint_function(layer, * curr_batch)
        outs.append(chunk_out)
    outs= torch.cat(outs, dim=0)
    outs= outs.reshape(*ex_bshapes, *outs.shape[1:])
    return outs
    


"""A collection of JAX utility functions for use in protein folding."""

# import torch


def final_init(config):
  if config.zero_init:
    return 'zeros'
  else:
    return 'linear'




def mask_mean(mask, value, dims=None, eps=1e-10):
  if dims is None:
    dims = list(range(len(value.shape)))

  broadcast_factor = 1.
  for axis_ in dims:
    value_size = value.size(axis_)
    mask_size = mask.size(axis_)
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      assert mask_size == value_size
  return torch.sum( mask *value, dim=dims) / (torch.sum(mask, dim=dims) * broadcast_factor + eps)


def moveaxis(data, source, destination):
  n_dims = len(data.shape)
  dims = [i for i in range(n_dims)]
  if source < 0:
    source += n_dims
  if destination < 0:
    destination += n_dims

  if source < destination:
    dims.pop(source)
    dims.insert(destination, source)
  else:
    dims.pop(source)
    dims.insert(destination, source)

  return data.permute(*dims)


  
