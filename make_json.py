import json
from collections import OrderedDict

file_data = OrderedDict()

file_data["input_shape"] = (3, 224, 224)
file_data["output_shape"] = (1, 1000)
file_data["batch_size"] = 1
file_data["model_path"] = "/Users/kimdonghyub/tvm-workspace/tvm_model_extractor/onnx_models/resnet101-v1-7.onnx"
file_data["optimization_level"] = 0
file_data["local"] = True
file_data["local_host"] = "llvm"
file_data["target_host"] = "llvm -mtriple=aarch64-conda-linux-gnu"
file_data["target_gpu"] = "opencl -device=mali"
file_data["rpc_ip_address"] = "192.168.1.189"
file_data["rpc_port"] = 9090


print(json.dumps(file_data, ensure_ascii=False,indent="\t"))

with open("arguments.json","w",encoding="utf-8") as make_file:
    json.dump(file_data, make_file, ensure_ascii=False, indent="\t")


