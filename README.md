This is a small repo demonstrating how one can create pytorch models using the python API and load these models in C++

This is mainly done following the tutorial at https://pytorch.org/tutorials/advanced/cpp_export.html

Here we cover several steps:

* Installing the module and dependencies
* Writing a pytorch model in python and saving it
* Loading that model and passing data through it on CPU and GPU.




# Installation:

### Conda environment
Run 

```
conda create --name pytorch_cpp
conda activate pytorch_cpp
pip install -r requirements.txt
```
### Libtorch installation
Install libtorch. Go to https://pytorch.org/get-started/locally/, select C++ and the version of CUDA which you have installed on your system. Download the package and unzip it ina directory of your choosing.

### Create a pytorch model

Run the python script:
```
python build_and_save_model.py 
```
this should generate the file `traced_resnet_model.pt` in the directory

### Build and make 
From the top level of the repo, Run
```
cmake -DCMAKE_PREFIX_PATH=~/path/to/libtorch/ . -B build
```
replacing `/path/to/libtorch` with the directory where you unzipped libtorch


### Run the c++ code

```
cd build
make
./load_torch_model ../traced_resnet_model.pt  
```

The output should look like 

```
CUDA is available! Training on GPU.
ok
Pushing ones through resnet...in C++! 
 2.6945
 2.7104
 2.8513
 2.9191
 2.8151
[ CPUFloatType{5} ]
Pushing zeros through resnet...in C++! 
 0.8088  0.1040  0.5983  0.9497  0.2337
 0.3928  0.0537  0.3828  0.9037  0.9774
 0.5262  0.2942  0.9835  0.9972  0.0503
 0.8150  0.1362  0.9310  0.6465  0.3904
 0.9942  0.0897  0.2324  0.5038  0.9950
[ CPUFloatType{5,5} ]
Pushing ones through resnet...in C++, on the gpu! 
 2.6945
 2.7104
 2.8513
 2.9191
 2.8151
[ CUDAFloatType{5} ]
```
voila!