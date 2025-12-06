import torch
import kornia.contrib as kc
import kornia.filters as kf

def find_centroids(image_mask):
    """
    Finds the centroids (center of weight) for each connected component in a batch of images.

    Args:
        image_mask (torch.Tensor): A binary input tensor of shape (B, 1, H, W) or (B, H, W).
    Returns:
        list: A list of tensors, where each tensor contains the (y, x) coordinates 
              of the centroid for the corresponding image in the batch.
    """
    if image_mask.dim() == 4:
        image_mask = image_mask[:, 0, :, :]
    
    labeled_image = kc.connected_components(image_mask.int()) #

    batch_centroids = []
    for b in range(labeled_image.shape[0]):
        img_labels = labeled_image[b]
        unique_labels = torch.unique(img_labels)
        unique_labels = unique_labels[unique_labels != 0]

        centroids = []
        for label in unique_labels:
            component_mask = (img_labels == label).float()
            
            area = torch.sum(component_mask)
            
            if area > 0:
                coords = torch.nonzero(component_mask)
                
                centroid_y = torch.sum(coords[:, 0]) / area
                centroid_x = torch.sum(coords[:, 1]) / area
                centroids.append(torch.tensor([centroid_y, centroid_x], device=image_mask.device))
        
        batch_centroids.append(torch.stack(centroids) if centroids else torch.tensor([], device=image_mask.device))

    return batch_centroids