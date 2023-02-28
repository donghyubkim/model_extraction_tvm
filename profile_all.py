from data_handling import model_selector, aggregated_result_writer, csv_merger_cleaner,remove_half,aggregate_data
from make_json import make_json

import tvm_profiler

run_iter = 20
input_shape = (3,224,224)
local = False
remove_half = True
path = "/Volumes/PATRIOT/onnx_models"
#path = "../onnx_models_int"
#path = "../onnx_models"


if remove_half: 
    run_iter *=2

onnx_model_list = model_selector(path)
onnx_cannot_compile_list = list()
for index,model in enumerate(onnx_model_list):
    print('progress: {}%'.format(((index+1)/len(onnx_model_list))*100)) # >> progress: 65.xx%

    try:
        profiler = tvm_profiler.tvm_profiler() #if out of this for loop (without initialization) we wait forever.
        make_json(model = model,path = path,input_shape=input_shape,local=local)
        profiler.compile()
        run_count = 1
        truncate_length = -(len(model.split('.')[-1]) +1) # to truncate .onnx from csv filename 

        for _ in range(run_iter): # run (run_iter) times 

            
            profiler.run()
            print("run {}".format(run_count))

            layer_information_dict = aggregate_data()
            layer_information_dict['label_model_name'] = model 

            aggregated_result_writer(model,layer_information_dict,run_count,truncate_dotonnx=truncate_length)

            run_count+=1
    except Exception as e:
        print(e)
        onnx_cannot_compile_list.append((model,e))
        continue


print(onnx_cannot_compile_list)

filename = "pred_model_trainable_data.csv"

csv_merger_cleaner(filename = filename,delete_aggregated_result_dir_files = False) # default arg: fillna0 = True 
# by filling NaN with 0, we can make unique layer as a feature.

if remove_half:
    remove_half(filename = filename)

