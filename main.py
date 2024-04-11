import torch
from torchvision import models

def count_operations(model, input_size=(3, 224, 224)):
    model.eval()
    inputs = torch.randn(1, *input_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = inputs.to(device)

    total_mults = 0
    total_adds = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            out_size = inputs.size(2) * inputs.size(3)
            total_mults += in_channels * out_channels * kernel_size * out_size
            total_adds += (in_channels * out_channels * kernel_size - 1) * out_size
        elif isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            total_mults += in_features * out_features
            total_adds += (in_features * out_features - 1)

    return total_mults, total_adds


if __name__ == '__main__':
    alexnet = models.alexnet()
    vgg19 = models.vgg19()
    resnet152 = models.resnet152()

    mults_alexnet, adds_alexnet = count_operations(alexnet)
    mults_vgg19, adds_vgg19 = count_operations(vgg19)
    mults_resnet152, adds_resnet152 = count_operations(resnet152)

    print("AlexNet - 乘法数量:", mults_alexnet, "加法数量:", adds_alexnet)
    print("VGG19 - 乘法数量:", mults_vgg19, "加法数量:", adds_vgg19)
    print("ResNet152 - 乘法数量:", mults_resnet152, "加法数量:", adds_resnet152)