import os 
import pandas as pd
import csv
from collections import defaultdict


def model_selector(path):
    #/directory/of/models
    
    onnx_model_list = os.listdir(path)
    target = ".onnx"
    onnx_model_list = leave_target_only(onnx_model_list,target)
    
    print('list of onnx models be profiled')
    onnx_model_list.sort()
    print(onnx_model_list)
    return onnx_model_list
def remove_white_space(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        lines = [line.replace(" ", "").replace("\t", "") for line in lines]

# save file with same filename
    with open(filename, "w") as file:
        file.writelines(lines)

def remove_white_space_all():
    path = "./aggregated_results"
    csv_to_be_merged_list = os.listdir(path)
    target = ".csv"
    
    csv_to_be_merged_list = leave_target_only(csv_to_be_merged_list,target)
    

    csv_file_path_list = list()

    for csv_file in csv_to_be_merged_list:
        csv_file_path = './aggregated_results/'+csv_file
        csv_file_path_list.append(csv_file_path)
    
    
    for path in csv_file_path_list:
        remove_white_space(path)
   

def csv_merger_cleaner(filename, fillna0 = True, delete_aggregated_result_dir_files = True): #it merges several model files into 1 csv file that can be trained in ML model.
    path = "./aggregated_results"
    csv_to_be_merged_list = os.listdir(path)
    target = ".csv"
    
    csv_to_be_merged_list = leave_target_only(csv_to_be_merged_list,target)
    

    csv_file_path_list = list()

    for csv_file in csv_to_be_merged_list:
        csv_file_path = './aggregated_results/'+csv_file
        csv_file_path_list.append(csv_file_path)
    
    print(csv_file_path_list)
    csv_base=pd.read_csv(csv_file_path_list[0]).astype('float64',errors = 'ignore')
    
    
    csv_base = csv_base.iloc[::2, :]
    print(csv_base)
    for path in csv_file_path_list[1:]: 
        csv_other = pd.read_csv(path).astype('float64',errors = 'ignore')
        
        
        csv_other = csv_other.iloc[0::2, :]
        print(csv_other)
        csv_base = pd.merge(csv_base, csv_other, how='outer')
    

    csv_base.fillna(0,inplace=fillna0)

    csv_base.to_csv(filename, encoding='utf-8',index = False)

    
    if delete_aggregated_result_dir_files:
        import glob
        files1 = glob.glob('./aggregated_results/**/*.csv', recursive=True)
        files2 = glob.glob('./profiling_result/**/*.csv', recursive=True)

        for f1,f2 in zip(files1,files2):
            try:
                os.remove(f1)
                os.remove(f2)
            except OSError as e:
                print("Error: %s : %s" % (f1, e.strerror))
                print("Error: %s : %s" % (f2, e.strerror))
    else:
        pass


def aggregated_result_writer(model,layer_information_dict,run_count,truncate_dotonnx):
    '''
    using aggregated result dictionary write it onto .csv file multiple times
    
    '''
    folder_name = './aggregated_results/result_{}.csv'.format(model[:truncate_dotonnx])

    with open(folder_name, 'a') as csvfile: #truncate .onnx in the csv filename
            writer = csv.DictWriter(csvfile, fieldnames=layer_information_dict.keys())
            if run_count == 1:
                writer.writeheader()
                writer.writerow(layer_information_dict)
            else:
                writer.writerow(layer_information_dict)
    
        

def leave_target_only(directory,target):
    new_directory = [related for related in directory if target in related]
    return new_directory

def remove_half(filename): #because of some unknown error that showing 0 s and nan % in the final aggregated result 
    # read the original csv file
    df = pd.read_csv(filename)
    # remove odd rows
    df = df.iloc[::2, :]
    # save the new csv file
    df.to_csv(filename+'_half_removed', index=False)
    

def aggregate_data() :
    '''
    Read from very first profiled data and aggregate them
    Return aggregated Dictionary as Dict[key: Aggregated layer name,value: summation of same layers]
    '''
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

        if not layer :
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
    
    
    
    return layer_dict



if __name__ == "__main__":
    #model_selector()
    #remove_white_space_all()
    csv_merger_cleaner(filename = "pred_model_trainable_data.csv",delete_aggregated_result_dir_files = False)

    '''
    layer_dict = feature_engineering()
    with open('aggregated_result.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=layer_dict.keys())
        #writer.writeheader()
        writer.writerow(layer_dict)
    '''