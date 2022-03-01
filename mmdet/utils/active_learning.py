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
        tensor = torch.concat( (tensor[:,:,None], 1 - tensor[:,:,None]), dim=2 ) # dimension ( N(images), N(bbox), 2 )
    
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
        return tensor.max(dim=1)[0]
    elif method == 'average':
        return tensor.mean(dim=1)
    elif method == 'sum':
        return tensor.sum(dim=1)



def select_images(method, tensor, n_sel, **kwargs):
    '''
    Select set of unlabelled images to be added to training set

    Arguments:
    
    method (str): name of method used to select images
    tensor (torch.tensor): output of aggregate_uncertainty, dimension ( N(images) )
    n_sel (int): number of images to select

    Returns:

    torch.tensor with index of selected images, dimension ( n_sel )
    '''

    if method == 'batch':
        return batch_selection(tensor, n_sel, **kwargs)
    elif method == 'maximum':
        return maximum_selection(tensor, n_sel)
    elif method == 'CoreSet':
        return core_set(tensor, n_sel, **kwargs)



def batch_selection(tensor, n_sel, **kwargs):
    '''
    Split randomly shuffled images in batches, compute aggregate score (sum) for each batch and select highest scores until n_sel is reached.

    Necessary keyword argument:

    batch_size (int): number of images per batch
    '''
    if (n_sel % kwargs['batch_size']) == 0:
        batch_size_sel = int(n_sel / kwargs['batch_size'])
    else:
        batch_size_sel = (n_sel // kwargs['batch_size']) + 1
    
    r = torch.randperm(tensor.shape[0])
    tensor_shuffle = tensor[r] # randomly shuffle images

    batch_list = tensor_shuffle.split(kwargs['batch_size'])
    batch_score_tensor = torch.tensor( [ b.sum().item() for b in batch_list ] )
    batch_argmax = batch_score_tensor.sort(descending=True)[1][:batch_size_sel]

    arg_sel = torch.concat( [r.split(kwargs['batch_size'])[ib] for ib in range(len(batch_list)) if ib in batch_argmax] )

    return arg_sel



def maximum_selection(tensor, n_sel):
    '''
    Select the n_sel images with the highest uncertainty.
    '''
    return tensor.sort(descending=True)[1][:n_sel]



def core_set(tensor, n_sel, **kwargs):
    '''
    k-center greedy, limited to pool set and with distance weighted by uncertainty
    '''
    # compute matrix of distances between images
    dist_mat = kwargs['embedding'] @ kwargs['embedding'].transpose(0, 1)
    diag = dist_mat.diag().reshape( (kwargs['embedding'].shape[0], 1) )
    dist_mat *= -2
    dist_mat += diag
    dist_mat += diag.transpose(0, 1)
    dist_mat = torch.sqrt(dist_mat)

    # choose first centroid randomly
    centroids = torch.zeros(tensor.shape[0], dtype=torch.bool)
    sel_idx = torch.randint( kwargs['embedding'].shape[0], (1,1) ).squeeze().item()
    centroids[sel_idx] = True
    
    # select centroids (= data to be labelled) iteratively
    for _ in range(n_sel-1):
        d = dist_mat[~centroids][:,centroids] # remove centroids from row and keep only centroids for columns
        unc = tensor[~centroids] # get uncertainty of images other than centroids
        
        mat_min = d.min(dim=1)[0] # get closest centroid for each image
        weighted_mat_min = mat_min * unc # weight distance to closest centroid by uncertainty on image
        sel_idx_ = weighted_mat_min.argmax().item() # select image with largest distance to closest centroid
        sel_idx = torch.arange(tensor.shape[0])[~centroids][sel_idx_] # correct index for already selected images/rows
        centroids[sel_idx] = True

    return torch.arange(tensor.shape[0])[centroids]
