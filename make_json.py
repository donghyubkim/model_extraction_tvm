def make_json(model,path,input_shape,local,opt_level,output_shape):

    import json
    from collections import OrderedDict

    file_data = OrderedDict()
    
    file_data["input_shape"] = input_shape
    file_data["output_shape"] = output_shape
    file_data["batch_size"] = 1
    
    file_data["model_path"] = path+'/'+model
    file_data["optimization_level"] = opt_level
    file_data["local"] = local
    file_data["local_host"] = "llvm"
    file_data["target_host"] = "llvm -mtriple=aarch64-conda-linux-gnu"
    file_data["target_gpu"] = "opencl -device=mali"
    file_data["rpc_ip_address"] = "192.168.1.189"
    file_data["rpc_port"] = 9090
    



    print(json.dumps(file_data, ensure_ascii=False,indent="\t"))

    with open("arguments.json","w",encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")
    
    


