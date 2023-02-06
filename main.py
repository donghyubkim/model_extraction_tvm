import os 

path = "../onnx_models"
onnx_model_list = os.listdir(path)
print(onnx_model_list)

for file in onnx_model_list:
    filepath = path + '/' + file
    print(file)