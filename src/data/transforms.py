import torchvision.transforms as transforms

def get_train_transforms(image_size=224):
    """
    Returns transformations for training data:
    - Resize
    - RandomHorizontalFlip
    - ColorJitter
    - ToTensor
    - Normalize (ImageNet stats)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_valid_transforms(image_size=224):
    """
    Returns transformations for validation/inference data:
    - Resize
    - ToTensor
    - Normalize
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
