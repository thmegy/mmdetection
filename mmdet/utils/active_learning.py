import torch


def estimate_uncertainty(method, tensor):
    '''
    Estimate classification uncertainty for a set of predicted bounding boxes.

    Arguments:
    
    method (str): name of method used to compute uncertainty
    tensor (torch.tensor): classification output of detection network, dimension ( N(images), N(bbox), N(classes) )

    Returns:

    torch.tensor with dimension ( N(images), N(bbox) )
    '''
    if tensor.ndim == 2: # only one class, if binary classification
        tensor = torch.concat( (tensor[:,:,None], 1 - a[:,:,None]), dim=2 ) # dimension ( N(images), N(bbox), 2 )
    
    if method == 'MarginSampling':
        return margin_sampling(tensor)
    elif method == 'Entropy':
        return entropy(tensor)
    elif method == 'VarRatio':
        return var_ratio(tensor)


def margin_sampling(tensor):
    '''
    Measure uncertainty as 1 - difference between probabilities of the two highest-ranking classes
    '''
    tensor_sorted = tensor.sort(dim=2, descending=True)[0] # sort class probabilities
    return 1 - (tensor_sorted[:,:,0] - tensor_sorted[:,:,1])



def entropy(tensor):
    '''
    Measure uncertainty as predictive entropy
    '''
    tensor_log = tensor.log()
    return (tensor*tensor_log).sum(dim=2)



def var_ratio(tensor):
    '''
    Measure uncertainty as 1 - probability of highest-ranking class
    '''
    return 1 - tensor.max(dim=2)[0]



def aggregate_uncertainty(method, tensor, weight=None):
    '''
    Aggregate uncertainties for individual bounding boxes into an overall uncertainty for an image.

    Arguments:
    
    method (str): name of method used to aggregate the uncertainty
    tensor (torch.tensor): output of estimate_uncertainty, dimension ( N(images), N(bbox) )
    weight (torch.tensor): weights to apply element-wise in aggregation, e.g. weight classification uncertainty by objectness uncertainty in YOLOv3

    Returns:

    torch.tensor with dimension ( N(images) )
    '''
    if weight is not None:
        tensor = tensor * weight
    
    if method == 'maximum':
        return tensor.max(dim=1)
    elif method == 'average':
        return tensor.mean(dim=1)
    elif method == 'sum':
        return tensor.sum(dim=1)
