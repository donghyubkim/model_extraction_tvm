from functions import model_selector
from make_json import make_json
import tvm_profiler 
from feature_engineer import feature_engineering
import csv

onnx_model_list = model_selector()
profiler = tvm_profiler.tvm_profiler
run_iter = 5
input_shape = (3,224,224)

for model in onnx_model_list:
    
    
    make_json(model,input_shape=input_shape)
    lib,remote_lib,dev,device_input_data = profiler.compile()
    
    run_count = 1
    for _ in range(run_iter): # run 5 times 
        
        profiler.run(lib,remote_lib,dev,device_input_data)
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
    
    




