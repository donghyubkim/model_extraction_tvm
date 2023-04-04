import numpy as np
from data_handling import model_selector, aggregated_result_writer, csv_merger_cleaner,remove_half,aggregate_data
import tvm
from tvm import rpc
from tvm.contrib.debugger import debug_executor

import json

class tvm_profiler_victim:

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

        local = args["local"]

        #specify target variable

        if local == True:
            remote = rpc.LocalSession()

        else:
            #connet to the remote runtime 
            #input board's local IP address.
            host = args["rpc_ip_address"]
            port = args["rpc_port"]
            remote= rpc.connect(host, port) #returns RPC session
            print("connecting to RPC server ...")

       
        path = '/home/donghyub/tvm_lib/lib.tar'
        self.remote_lib = remote.load_module(path) # Runtime
        print("successfully loaded model library")


        #specify device
        if local == True:
            self.dev = remote.cpu(0)
        else:
            self.dev = remote.cl(0)


        #make device specified input
        self.device_input_data = tvm.nd.array(data, self.dev)
        
        self.profiler = debug_executor.create(self.remote_lib["get_graph_json"](), self.remote_lib, self.dev)
        

    def run(self) -> None:
        ## profiler run1
        
        report = self.profiler.profile(data=self.device_input_data)
        report_table_full = report.csv()
        layer_information_dict = aggregate_data(victim_profile_mode=True)# itself doesn't have label. just like x data.
        aggregated_result_writer(model = None, layer_information_dict = layer_information_dict ,run_count = 1,truncate_dotonnx=None,victim_profile_mode=True)
        # writing to csv file 
        with open("./profiling_result/victim_result_full.csv", 'w') as out:
            out.write(report_table_full) #we are using this one to aggregate


if __name__ == "__main__":
    t = tvm_profiler_victim()
    t.compile()
    t.run()
