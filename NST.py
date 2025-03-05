import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
from models.definitions.vgg19 import Vgg19

# Constants for normalization
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

# Load image from path
def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path not found: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB
    if target_shape:
        img = cv.resize(img, (target_shape, target_shape), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    return img

# Prepare image tensor for NST
def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    return transform(img).to(device).unsqueeze(0)

# Save tensor image to disk
def save_image(img_tensor, img_path):
    img = np.moveaxis(img_tensor.squeeze(0).cpu().detach().numpy(), 0, 2)
    # Denormalize the image
    img = np.clip(img + np.array(IMAGENET_MEAN_255).reshape((1, 1, 3)), 0, 255).astype('uint8')
    cv.imwrite(img_path, img[:, :, ::-1])  # Convert RGB back to BGR

# Load the VGG19 model for feature extraction
def prepare_model(device):
    model = Vgg19(requires_grad=False, show_progress=True)
    return model.to(device).eval(), model.content_feature_maps_index, model.style_feature_maps_indices

# Compute Gram matrix for style representation
def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    return features.bmm(features.transpose(1, 2)) / (ch * h * w)

# Compute content and style loss
def build_loss(neural_net, optimizing_img, target_representations, content_index, style_indices, config):
    target_content = target_representations[0]
    target_style = target_representations[1]
    current_features = neural_net(optimizing_img)
    # Content loss: MSE between target content and current content features
    content_loss = torch.nn.MSELoss()(target_content, current_features[content_index].squeeze(0))
    # Style loss: MSE between target style (Gram matrices) and current style Gram matrices
    style_loss = 0
    current_style = [gram_matrix(x) for cnt, x in enumerate(current_features) if cnt in style_indices]
    for gt, cur in zip(target_style, current_style):
        style_loss += torch.nn.MSELoss()(gt, cur)
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss
    return total_loss

# Main NST function
def neural_style_transfer(config):
    # Build full paths for content and style images
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    # Output file path (final combined image)
    output_file = os.path.join(config['output_img_dir'],
                               f"combined_{config['content_img_name'].split('.')[0]}_{config['style_img_name'].split('.')[0]}.jpg")
    
    os.makedirs(config['output_img_dir'], exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    # Initialize optimizing image as a clone of the content image
    optimizing_img = Variable(content_img.clone(), requires_grad=True)
    
    neural_net, content_index, style_indices = prepare_model(device)
    
    # Get target representations
    target_content = neural_net(content_img)[content_index].squeeze(0)
    # Compute Gram matrices for the style image features
    target_style = [gram_matrix(x) for cnt, x in enumerate(neural_net(style_img)) if cnt in style_indices]
    target_representations = [target_content, target_style]
    
    optimizer = LBFGS([optimizing_img], max_iter=config['num_of_iterations'], line_search_fn='strong_wolfe')
    cnt = 0

    # Closure function for LBFGS optimization
    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss = build_loss(neural_net, optimizing_img, target_representations, content_index, style_indices, config)
        total_loss.backward()
        print(f'Iteration {cnt:03}, Total Loss: {total_loss.item():.4f}')
        # Save image every 50 iterations
        if cnt % 50 == 0:
            save_image(optimizing_img, output_file)
            print(f"Image saved at iteration {cnt}: {output_file}")
        cnt += 1
        return total_loss

    optimizer.step(closure)
    # Save final image
    save_image(optimizing_img, output_file)
    print(f"Final image saved: {output_file}")
    return output_file

# Configuration
PATH = r"C:\Users\rushi_a87oqn1\Desktop\Neural-Style-Transfer\data"
optimization_config = {
    'content_img_name': "c1.jpg",
    'style_img_name': "s1.jpg",
    'height': 400,
    'content_weight': 10000.0,
    'style_weight': 1000.0,
    'num_of_iterations': 500,  # You can adjust this number
    'content_images_dir': os.path.join(PATH, 'content-images'),
    'style_images_dir': os.path.join(PATH, 'style-images'),
    'output_img_dir': os.path.join(PATH, 'output-images')
}

# Run NST
results_path = neural_style_transfer(optimization_config)
