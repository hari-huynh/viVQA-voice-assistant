from huggingface_hub import hf_hub_download
import zipfile
import json
from tqdm import tqdm
import os


def convert_json(qa_dict,img_dict):
  llava_data_list = []

  for _, qa in tqdm(qa_dict.items()):
      img_id = str(qa['image_id'])
      image = img_dict[img_id]
      ques = qa['question']
      ans = qa['answer']

      llava_data_list.append(
          {
              "id": img_id,
              "image": image,
              "conversations": [
                  {
                      "from": "human",
                      "value": ques
                  },
                  {
                      "from": "gpt",
                      "value": ans
                  }
              ]
          }
      )
  return llava_data_list

def load_image_zip(file_name, zip_file_path,extract_folder_path):

  hf_hub_download(
    repo_id='uitnlp/OpenViVQA-dataset',
    repo_type='dataset',
    filename=file_name,# 'train-images.zip',
    local_dir='.'
  )

  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)


def load_json(file_name,file_path, file_name_json):
  hf_hub_download(
      repo_id='uitnlp/OpenViVQA-dataset',
      repo_type='dataset',
      filename=file_name, #'vlsp2023_train_data.json',
      local_dir='.'
  )
  with open(file_path, 'r') as file:
    data = json.load(file)
  img_dict = data['images']
  qa_dict = data['annotations']

  llava_data_list=convert_json(qa_dict,img_dict)

  json_output_path = os.path.join('./', file_name_json)
  with open(json_output_path, 'w', encoding="utf8") as json_file:
      json.dump(llava_data_list, json_file, indent=4, ensure_ascii=False)



####### train_image
file_name='train-images.zip'
zip_file_path = './train-images.zip'
extract_folder_path = './train-images'
load_image_zip(file_name, zip_file_path,extract_folder_path)


######## dev_image
file_name='dev-images.zip'
zip_file_path = './dev-images.zip'
extract_folder_path = './dev-images'
load_image_zip(file_name, zip_file_path,extract_folder_path)


####### train_json

file_name='vlsp2023_train_data.json'
file_path = './vlsp2023_train_data.json'
file_name_json='train_dataset.json'
load_json(file_name,file_path, file_name_json)

###### dev_json

file_name='vlsp2023_dev_data.json'
file_path = './vlsp2023_dev_data.json'
file_name_json='eval_dataset.json'
load_json(file_name,file_path, file_name_json)