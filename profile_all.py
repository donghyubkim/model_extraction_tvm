from functions import model_selector
from make_json import make_json
from tvm_profiler import tvm_profiler, profiler_run
from feature_engineer import feature_engineering
import csv

onnx_model_list = model_selector()

for model in onnx_model_list:

    input_shape = (3,224,224)
    make_json(model,input_shape=input_shape)
    lib,remote_lib,dev,device_input_data = tvm_profiler()
    
    run_count = 1
    for _ in range(5): # run 5 times 
        profiler_run(lib,remote_lib,dev,device_input_data)
        print("run {}".format(run_count))
        run_count+=1
        layer_information_dict = feature_engineering()
        with open('aggregated_result.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=layer_information_dict.keys())
            if run_count == 1:
                writer.writeheader()
                writer.writerow(layer_information_dict)
            else:
                writer.writerow(layer_information_dict)
    
    




