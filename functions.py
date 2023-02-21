import os 
import pandas as pd
import csv

def model_selector(path):
    #/directory/of/models
    
    onnx_model_list = os.listdir(path)
    print('list of onnx models be profiled')
    target = ".onnx"
    onnx_model_list = leave_target_only(onnx_model_list,target)
    print(onnx_model_list)
    

    return onnx_model_list

def csv_merger_cleaner(filename, fillna0 = True, delete_aggregated_result_dir_files = True): #it merges several model files into 1 csv file that can be trained in ML model.
    path = "./aggregated_results"
    csv_to_be_merged_list = os.listdir(path)
    target = ".csv"
    
    csv_to_be_merged_list = leave_target_only(csv_to_be_merged_list,target)
    

    csv_file_path_list = list()

    for csv_file in csv_to_be_merged_list:
        csv_file_path = './aggregated_results/'+csv_file
        csv_file_path_list.append(csv_file_path)
    
    
    csv_base=pd.read_csv(csv_file_path_list[0])
    csv_base = csv_base.iloc[2::2, :]
    
    for path in csv_file_path_list[1:]: 
        csv_other = pd.read_csv(path)
        csv_other = csv_other.iloc[2::2, :]
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
    






if __name__ == "__main__":
    #model_selector()
    csv_merger_cleaner(filename = "pred_model_trainable_data.csv",delete_aggregated_result_dir_files = False)
