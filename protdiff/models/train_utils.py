import torch

def mask_loss(loss_masks, loss_dict):
    """ items need to be mask is filled with True, else is False
    """
    assert isinstance(loss_masks, torch.Tensor)
    loss_masks = 1 - loss_masks.int()
    
    def _apply(x):
        if torch.is_tensor(x):
            len_shape = len(x.shape)
            [loss_masks.unsqueeze_(1) for _ in range(len_shape-1)]
            return loss_masks * x
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(loss_dict)



def merge_all(batch_dict):
    single_dict, pair_dict, frame_dict = {}, {}, {}
    for k, v in batch_dict.items():
        if 'single' in k:
            single_dict.update({k: v})
        elif 'pair' in k:
            pair_dict.update({k: v})
        elif 'traj_backbone_frame' in k:
            frame_dict.update({k: v})
        else:
            continue

    return single_dict, pair_dict, frame_dict