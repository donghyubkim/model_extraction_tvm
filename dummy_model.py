import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnx.utils
from onnx import checker,shape_inference

def generate_dummy_inserted(num_of_dummy):
    # Load the ResNet18 model from the PyTorch Model Zoo
    model = models.resnet18(pretrained=True)

    # Freeze all the existing layers in the ResNet18 model
    for param in model.parameters():
        param.requires_grad = False

    # Insert new convolutional layers after the first convolutional layer in the ResNet18 model
    new_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    if num_of_dummy == 1:
    # 1 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv)
        model.layer4= nn.Sequential(new_conv,model.layer4)
    elif num_of_dummy ==2:
    #2 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,model.layer4)
    elif num_of_dummy == 3:
    #3 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 4:
    #4 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 5:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 6:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 7:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 8:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 9:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 10:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 11:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 12:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 13:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 14:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
    elif num_of_dummy == 15:
    #5 dummies
        #model.conv1 = nn.Sequential(model.conv1, new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv)
        model.layer4= nn.Sequential(new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,new_conv,model.layer4)
        
    else:pass



    # Print the updated model architecture
    print(model)

    model.eval()

    # Create a dummy input tensor to export the model
    input_tensor = torch.randn(1, 3, 224, 224)


    # Export the model to ONNX format
    onnx_model = torch.onnx.export(model, input_tensor, "resnet18_extended.onnx")

    # Load the exported ONNX model
    onnx_model = onnx.load("resnet18_extended.onnx")


    # Verify the ONNX model
    checker.check_model(onnx_model)

    # Convert the ONNX model to the latest version
    #onnx_model = version_converter.convert_version(onnx_model, 11)

    # Update the model dimensions to remove dynamic axes
    #onnx_model = update_model_dims(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)

    # Save the ONNX model to a file
    onnx.save(onnx_model, "dummy_inserted_models/resnet18_dummy{}.onnx".format(num_of_dummy))
if __name__ == "__main__":
    for num in range(10,16):
        generate_dummy_inserted(num)