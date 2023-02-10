import os 
import pandas as pd
import csv

def model_selector():
    #/directory/of/models
    path = "../onnx_models"
    onnx_model_list = os.listdir(path)
    print('list of onnx models be profiled')
    print(onnx_model_list)
    
    onnx_model_list.remove(".DS_Store")

    return onnx_model_list

def csv_merger(filename, fillna0 = True, delete_aggregated_result_dir_files = True): #it merges several model files into 1 csv file that can be trained in ML model.
    path = "./aggregated_results"
    csv_to_be_merged_list = os.listdir(path)
    
    csv_to_be_merged_list.remove(".DS_Store")

    csv_file_path_list = list()

    for csv_file in csv_to_be_merged_list:
        csv_file_path = './aggregated_results/'+csv_file
        csv_file_path_list.append(csv_file_path)
    
    
    csv_base=pd.read_csv(csv_file_path_list[0])

    for path in csv_file_path_list[1:] : 
        csv_other = pd.read_csv(path)
        csv_base = pd.merge(csv_base, csv_other, how='outer')
    

    csv_base.fillna(0,inplace=fillna0)

    csv_base.to_csv(filename, encoding='utf-8')

    
    if delete_aggregated_result_dir_files:
        import glob
        files = glob.glob('./aggregated_results/**/*.csv', recursive=True)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
    else:
        pass


def aggregated_result_writer(model,layer_information_dict,run_count,truncate_dotonnx):
    with open('./aggregated_results/result_{}.csv'.format(model[:truncate_dotonnx]), 'a') as csvfile: #truncate .onnx in the csv filename
            writer = csv.DictWriter(csvfile, fieldnames=layer_information_dict.keys())
            if run_count == 1:
                writer.writeheader()
                writer.writerow(layer_information_dict)
            else:
                writer.writerow(layer_information_dict)

if __name__ == "__main__":
    #model_selector()
    csv_merger(filename = "pred_model_trainable_result.csv")