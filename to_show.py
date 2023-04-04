import numpy as np
import tvm.relay as relay
import tvm
from tvm import rpc

from tvm.contrib.debugger import debug_executor

remote= rpc.connect(ip_address, port) #get runtime object remotely via RPC server
remote_lib = remote.load_module("path/to/model_lib") #pass model path in target device
dev = remote.cl(0)

device_input_data = tvm.nd.array(data,dev)
profiler = debug_executor.create(lib.get_graph_json(), remote_lib, dev)  # from here, if we would't be able to get graph json. so dummy json could be a way.
report = profiler.profile(data=device_input_data)
report_table_full = report.csv()