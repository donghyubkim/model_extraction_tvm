from functions import model_selector, aggregated_result_writer, csv_merger_cleaner
from make_json import make_json
from feature_engineer import feature_engineering
import tvm_profiler




run_iter = 40
input_shape = (3,224,224)
int_mod = True #if you want to profile integer models make it True
local = True
onnx_model_list = model_selector(int_mod)

for model in onnx_model_list:
    
    
    profiler = tvm_profiler.tvm_profiler() #if out of this for loop (without initialization) we wait forever.
    make_json(model,input_shape=input_shape,local=local,int_mod=int_mod)
    profiler.compile()
    
    run_count = 1
    truncate_length = -(len(model.split('.')[-1]) +1) # to truncate .onnx from csv filename 

    for _ in range(run_iter): # run 5 times 
        
        profiler.run()
        print("run {}".format(run_count))
        
        layer_information_dict = feature_engineering()
        layer_information_dict['label_model_name'] = model 

        aggregated_result_writer(model,layer_information_dict,run_count,truncate_dotonnx=truncate_length,int_mod = int_mod)

        run_count+=1

if not int_mod: 
    filename = "pred_model_trainable_data.csv"
else:
    filename = "pred_model_trainable_data_int8.csv"
csv_merger_cleaner(filename = filename,delete_aggregated_result_dir_files = False) # default arg: fillna0 = True 
# by filling NaN with 0, we can make unique layer as a feature.


