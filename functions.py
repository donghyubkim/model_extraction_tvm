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

def csv_merger(): #it merges several model files into 1 csv file that can be trained in ML model.
    path = "./aggregated_results"
    csv_to_be_merged_list = os.listdir(path)
    
    csv_to_be_merged_list.remove(".DS_Store")

    csv_file_path_list = list()

    for csv_file in csv_to_be_merged_list:
        csv_file_path = './aggregated_results/'+csv_file
        csv_file_path_list.append(csv_file_path)
    
    
    csv_base=pd.read_csv(csv_file_path_list[0] )

    for path in csv_file_path_list[1:] : 
        csv_other = pd.read_csv(path)
        csv_base = pd.merge(csv_base, csv_other, how='outer')
    
    
    csv_base.to_csv('final_result.csv', encoding='utf-8')



    '''
    data1 = pd.read_csv('datasets/loan.csv')
    data2 = pd.read_csv('datasets/borrower.csv')
    output4 = pd.merge(data1, data2, 
                        on='LOAN_NO', 
                        how='outer')
    '''
def aggregated_result_writer(model,layer_information_dict,run_count,truncate_length):
    with open('./aggregated_results/result_{}.csv'.format(model[:truncate_length]), 'a') as csvfile: #truncate .onnx in the csv filename
            writer = csv.DictWriter(csvfile, fieldnames=layer_information_dict.keys())
            if run_count == 1:
                writer.writeheader()
                writer.writerow(layer_information_dict)
            else:
                writer.writerow(layer_information_dict)

if __name__ == "__main__":
    #model_selector()
    csv_merger()