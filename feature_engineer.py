
#import pandas as pd
import csv
from collections import defaultdict

def feature_engineering():
    data_list = list()
    aggregated_layer_list = list()
    f = open('./profiling_result/result_full.csv','r',encoding="utf-8")
    reader = csv.reader(f)
    remove_set = ("")
    #print("From")
    #df = pd.read_csv('./profiling_result/result_full.csv')
    #print(df.head())
    for layer in reader:
        layer = [data for data in layer if data not in remove_set] 
        #print(layer)  

        if layer == []:
            continue
        elif 'tvmgen' in layer[5]: 
            layer[5] = layer[5].split("_")[3:] #get rid of tvmgen_default_fused 
            if layer[5][-1].isnumeric(): #get rid of number if ends with number
                layer[5].pop()
            else:
                pass #if doesn't end with a number, leave it
            
            layer[5] = "".join(layer[5]) #make str again
            if layer[5] not in aggregated_layer_list:
                aggregated_layer_list.append(layer[5])
            data_list.append(layer[1:7])
        
        else:
            continue
    #print(aggregated_layer_list)
    #print(data_list)

    #dictionary init. otherwise keyerror
    layer_dict = defaultdict(float)
    
    for data in data_list:
        # str -> int or float
        layer_duration = float(data[2].replace(",","")) #duration of a single layer
        layer_percentage = float(data[5]) #percentage of a single layer
        layer_count = int(data[0]) #count of a single layer, which is 1 
        #adding

        
        layer_name = data[4]
        layer_dict['aggregated_duration_'+ layer_name] += layer_duration
        layer_dict['aggregated_percentage_'+ layer_name] += layer_percentage
        layer_dict['aggregated_count_'+ layer_name] += layer_count

        layer_dict['total_duration'] += layer_duration
        layer_dict['total_percentage'] += layer_percentage 
        layer_dict['total_count'] += layer_count
    
    
    #for key in layer_dict_keys:
        #print(key)
        #print(round(layer_dict[key],3))
    return layer_dict

if __name__ == "__main__":
    layer_dict = feature_engineering()
    with open('aggregated_result.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=layer_dict.keys())
        #writer.writeheader()
        writer.writerow(layer_dict)