from functions import model_selector, aggregated_result_writer, csv_merger
from make_json import make_json
from feature_engineer import feature_engineering
import tvm_profiler


onnx_model_list = model_selector()
profiler = tvm_profiler.tvm_profiler()
run_iter = 5
input_shape = (3,224,224)

for model in onnx_model_list:
    
    
    
    make_json(model,input_shape=input_shape)
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


csv_merger(filename = "pred_model_trainable_result.csv") # default arg: fillna0 = True 
# by filling NaN with 0, we can make unique layer as a feature.


