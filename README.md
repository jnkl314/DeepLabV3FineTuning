# DeepLabV3FineTuning
Semantic Segmentation : Multiclass fine tuning of DeepLabV3 with PyTorch

The code in this repository performs a fine tuning of DeepLabV3 with PyTorch for multiclass semantic segmentation.

## Result Preview
Random result on a test image (not in dataset) </br>
<img src="./pictures/1426_image.png" width="240" height="135"><img src="./pictures/1426_segmentation.png" width="240" height="135"><img src="./pictures/1426_both.png" width="240" height="135">

## Requirements
Basic dependencies are PyTorch 1.4.0 and torchvision 0.5.0.</br>
I used a conda virtual env, where I installed the following packages :
```bash
conda install -c conda-forge -c pytorch python=3.7 pytorch torchvision cudatoolkit=10.1 opencv numpy pillow
```

## Dataset
I created a dataset from my own personal skydiving pictures.</br>
Around 500 images were gathered and annotated using the excellent tool CVAT : https://github.com/opencv/cvat </br>
<img src="./pictures/screenshot_cvat.png" width="640" height="360"></br>
/!\ On this repo, I only uploaded a few images in ![./sample_dataset](./sample_dataset) as to give an idea of the format I used.</br>
I wrote a ![script](./sample_dataset/convert_cvat_xml_to_label_image.py) to easily convert one of the XML export types (LabelMe ZIP 3.0 for images) of CVAT into label images</br>
There are 5 classes in my example: <br/>
* No-label : 0
* Person : 1
* Airplane : 2
* Ground : 3
* Sky : 4

## How to run training
Once you replace sample_data with your own dataset :
```bash
python sources/main_training.py ./sample_dataset ./training_output --num_classes 5 --epochs 100 --batch_size 16 --keep_feature_extract
```

TBC
