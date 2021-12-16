# F21 ROB535 Self-Driving Cars Image Classification

Given images and point clouds data, which contains the labels of 3-D bounding box information of objects, we created a model based on Yolov1 and obtained the accuracy of 55% after 140 iterations of training.

## Description

Firstly, we tried to preprocess our data by compressing images and converting 3-d bounding boxes into 2d bounding boxes. Then we created a model, which based on Yolov1, to classify our image.  The model we used can be found in the *model.py*. We defined our loss in the *loss.py*. We trained our neural network using Google Colab, which can be found in the *Test.ipynb*. Then we can output our result by running *results.ipynb*.

## Getting Started

### Dependencies

* The data is mainly processed in Ubuntu 18.04.
* torch
* Google Colab

### Installing

* You can totally run the program using Google Colab.

### Executing program

* Working Directory Tree

  ```
  ├── checkpoints
  ├── data
  │   ├── test
  │   └── trainval
  ├── dataset.py
  ├── extract_info.py
  ├── loss.py
  ├── model.py
  ├── preprocess.py
  ├── README.md
  ├── result.ipynb
  ├── result.py
  ├── Test.ipynb
  ├── train.py
  └── utils
      ├── transform.py
      └── utils.py
  ```

- Firstly you can download the data through this [link](https://drive.google.com/drive/folders/15LPTXADcZGv0ZE262yqdwFHDTnP_R_Bx). 
- Run *extract_info.py* to generate bbox.csv file for trianning.

	```bash
	python extract_info.py
	```

- Then run

  ```bash
  python preprocess.py
  ```

  to compress the image into 448 * 448.

* delete all *.bin files to keep our data small and clean by running:

  ```bash
  rm data/**/**.bin
  ```

* Generate rob535-data.zip by compressing our data directory.

* Open *Test.ipynb*  and run commands in the file to train our neural network in Google Colab.

* Open *result.ipynb*  and run commands to generate our output in Google Colab.

## Further Developments

- We could use a more suitable model rather than yolov1, which is more focused on object detection rather than image classification.


## Authors

Contributors names and contact info

- [@Shawn Wang](https://github.com/ShawnKing98)

- [@Xiaofeng Lin](https://github.com/potBagMeat)
- @Can Cui

## Reference

- [Yolov1 paper](https://arxiv.org/abs/1506.02640)

