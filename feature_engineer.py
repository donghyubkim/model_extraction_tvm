
import pandas as pd
import csv
data_list = list()
aggregated_layer_list = list()
f = open('./profiling_result/result_neat.csv','r',encoding="utf-8")
reader = csv.reader(f,delimiter=" ")
remove_set = ("")
print("From")
df = pd.read_csv('./profiling_result/result_neat.csv')
print(df.head())
for layer in reader:
    layer = [data for data in layer if data not in remove_set]   
    
    if layer == []:
        continue
    elif 'tvmgen' in layer[0]:
        layer[0] = layer[0].split("_")[3:] #get rid of tvmgen_default_fused 
        if layer[0][-1].isnumeric(): #get rid of number if ends with number
            layer[0].pop()
        else:
            pass #if doesn't end with a number, leave it
        
        layer[0] = "".join(layer[0]) #make str again
        if layer[0] not in aggregated_layer_list:
            aggregated_layer_list.append(layer[0])
        data_list.append(layer[:5])
    elif layer[0] == 'Total':
        Total = layer
    elif layer[0] == 'Sum':
        Sum = layer
    else:
        continue
#print(aggregated_layer_list)
#print(data_list)

#dictionary init. otherwise keyerror
layer_dict = dict()
for layer in aggregated_layer_list: 
    layer_dict["aggregated_duration_"+layer] = 0
    layer_dict["aggregated_percentage_"+layer] = 0
    layer_dict["aggregated_count_"+layer] = 0    
layer_dict_keys = layer_dict.keys()
for data in data_list:
    # str -> int or float
    data[1] = float(data[1].replace(",",""))
    data[2] = float(data[2])
    data[4] = int(data[4])
    #adding
    layer_dict['aggregated_duration_'+ data[0]] += data[1]
    layer_dict['aggregated_percentage_'+ data[0]] += data[2]
    layer_dict['aggregated_count_'+ data[0]] += data[4]
print("To")
print(Total)
print(Sum)
for key in layer_dict_keys:
    print(key)
    print(round(layer_dict[key],3))

