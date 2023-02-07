
#import pandas as pd
import csv

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
    layer_dict = dict()
    for layer in aggregated_layer_list: 
        layer_dict["aggregated_duration_"+layer] = 0
        layer_dict["aggregated_percentage_"+layer] = 0
        layer_dict["aggregated_count_"+layer] = 0   

    layer_dict['total_duration'] = 0 
    layer_dict['total_percentage'] = 0 
    layer_dict['total_count'] = 0 
    layer_dict_keys = layer_dict.keys()
    for data in data_list:
        # str -> int or float
        data[2] = float(data[2].replace(",","")) #duration of a single layer
        data[5] = float(data[5]) #percentage of a single layer
        data[0] = int(data[0]) #count of a single layer, which is 1 
        #adding

        layer_dict['aggregated_duration_'+ data[4]] += data[2]
        layer_dict['aggregated_percentage_'+ data[4]] += data[5]
        layer_dict['aggregated_count_'+ data[4]] += data[0]

        layer_dict['total_duration'] += data[2]
        layer_dict['total_percentage'] += data[5] 
        layer_dict['total_count'] += data[0] 
    
    
    for key in layer_dict_keys:
        print(key)
        print(round(layer_dict[key],3))
    return layer_dict

if __name__ == "__main__":
    layer_dict = feature_engineering()
    with open('aggregated_result.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=layer_dict.keys())
        #writer.writeheader()
        writer.writerow(layer_dict)