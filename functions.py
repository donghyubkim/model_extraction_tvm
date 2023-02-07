import os 




def model_selector():
    #/directory/of/models
    path = "../onnx_models"
    onnx_model_list = os.listdir(path)
    print(onnx_model_list)
    
    onnx_model_list.remove(".DS_Store")

    return onnx_model_list


if __name__ == "__main__":
    model_selector()