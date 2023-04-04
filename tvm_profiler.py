import numpy as np
import tvm.relay as relay
import tvm
from tvm import rpc
from tvm.contrib import utils
import onnx
from tvm.contrib.debugger import debug_executor

import json

class tvm_profiler:

    def __init__(self) -> None:
        self.lib = None
        self.remote_lib = None
        self.dev = None
        self.device_input_data = None

    def compile(self) -> None:

        with open ("arguments.json", "r") as f:
            args = json.load(f)

        #data preparation
        image_shape = args["input_shape"]
        data_shape = [args["batch_size"],] + image_shape
        #output_shape = args["output_shape"]
        data = np.random.uniform(-1, 1, size=data_shape).astype("float64")



        #onnx model preparation

        #put /path/to/your/model/
        model_path = args["model_path"]
        onnx_model = onnx.load(model_path)

        #Convert a ONNX model into an equivalent Relay Function 
        
        #if 'densenet' in model_path or 'squeezenet' in model_path: #because of maybe some bug with onnx model itself. 
        #    input_name = 'data_0'
        #else:
        input_name = [onnx_model.graph.input[0]][0].name

            
        #print([onnx_model.graph.input[0]][0])
        #print(input_name)
        #input_name = "input"
        shape_dict = {input_name: data_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict) #returns relay IR module for compilation

        #from doc:
        #Import: The frontend component ingests a model into an IRModule, 
        #which contains a collection of functions that internally represent the model.

        #print(mod.astext(show_meta_data=False)) #show onnx model layers



        # if not in same place with Mali board, make local True so we can test it in our home.
        # local True makes use our host machine cpu (in my case, macbook)
        local = args["local"]

        #specify target variable
        if local == True:
            target = args["local_host"]
        else:
            #remote board cpu information 
            target_host = args["target_host"]
            target = tvm.target.Target(args["target_gpu"], host=target_host)


        with tvm.transform.PassContext(opt_level=args["optimization_level"]): # optimization level from 0 to 3

            #Helper function that builds a Relay function to module that runs on TVM graph executor.
            self.lib = relay.build(mod, target=target,params=params) 
            #returns factory_module – The runtime factory for the TVM graph executor.

            #from doc:
            #Target Translation: The compiler translates(codegen) the IRModule to an executable format specified by the target. 
            #The target translation result is encapsulated as a runtime.
            #Module that can be exported, loaded, and executed on the target runtime environment.   

        if local == True:
            remote = rpc.LocalSession()

        else:
            #connet to the remote runtime 
            #input board's local IP address.
            host = args["rpc_ip_address"]
            port = args["rpc_port"]
            remote= rpc.connect(host, port) #returns RPC session
            print("connecting to RPC server ...")

        # save the lib at a local temp folder
        # uploading model library(binary) to remote runtime
        print("loading generated lib to runtime ...")
        temp = utils.tempdir()
        path = temp.relpath("lib.tar") #Actual path
        self.lib.export_library(path)
        remote.upload(path)

        self.remote_lib = remote.load_module("lib.tar") # Runtime
        #tvm.runtime.Module encapsulates the result of compilation.
        #A runtime.Module contains a GetFunction method to obtain PackedFuncs by name.
        print("successfully loaded model library")


        #specify device
        if local == True:
            self.dev = remote.cpu(0)
        else:
            self.dev = remote.cl(0)


        #make device specified input
        self.device_input_data = tvm.nd.array(data, self.dev)
        
        self.profiler = debug_executor.create(self.lib.get_graph_json(), self.remote_lib, self.dev)
        

    def run(self) -> None:
        ## profiler run1
        
        report = self.profiler.profile(data=self.device_input_data)
        report_table_full = report.csv()

        # writing to csv file 
        with open("./profiling_result/result_full.csv", 'w') as out:
            out.write(report_table_full) #we are using this one to aggregate
        #with open("./profiling_result/result_neat.csv", 'w') as out:
            #out.write(report_table_neat) #this one, tried, but doesn't work for some models (ex googlenet)



        ''' #running without profiling

        from tvm.contrib import graph_executor
        #run general inference and get output
        #make module that runs remote    
        module = graph_executor.GraphModule(remote_lib["default"](dev))
        module.set_input("data", device_input_data)
        module.run()
        #so in this code, we run twice. profiler and just module run.
        #get output
        out = module.get_output(0)
        # get top1 result
        top1 = np.argmax(out.numpy())
        print("TVM prediction top-1: {}".format(top1))
        '''
        

if __name__ == "__main__":
    t = tvm_profiler()
    t.compile()
    t.run()
