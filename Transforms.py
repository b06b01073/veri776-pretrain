from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 256


def get_training_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image to IMAGE_SIZE*IMAGE_SIZE
        transforms.ToTensor(),          
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(scale=(0.02, 0.4)), # BoT setting
    ])

    return transform


def get_test_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image to IMAGE_SIZE*IMAGE_SIZE
        transforms.ToTensor(),          
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transform