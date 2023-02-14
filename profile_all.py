from functions import model_selector, aggregated_result_writer, csv_merger_cleaner
from make_json import make_json
from feature_engineer import feature_engineering
import tvm_profiler


onnx_model_list = model_selector()

run_iter = 40
input_shape = (3,224,224)
int_mod = False #if you want to profile integer model make it True

for model in onnx_model_list:
    
    
    profiler = tvm_profiler.tvm_profiler() #if out of this for loop (without initialization) we wait forever.
    make_json(model,input_shape=input_shape, int_mod=int_mod)
    profiler.compile()
    
    run_count = 1
    truncate_length = -(len(model.split('.')[-1]) +1) # to truncate .onnx from csv filename 

    for _ in range(run_iter): # run 5 times 
        
        profiler.run()
        print("run {}".format(run_count))
        
        layer_information_dict = feature_engineering()
        layer_information_dict['label_model_name'] = model 

        aggregated_result_writer(model,layer_information_dict,run_count,truncate_dotonnx=truncate_length)

        run_count+=1


csv_merger_cleaner(filename = "pred_model_trainable_data.csv",delete_aggregated_result_dir_files = False) # default arg: fillna0 = True 
# by filling NaN with 0, we can make unique layer as a feature.


