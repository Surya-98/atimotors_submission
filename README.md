# Atimotors Assignment
Re-id(Re-identification) model and quantization

The model and weights were used from the following repository: \
https://github.com/layumi/Person_reID_baseline_pytorch.git \
Model - Densnet101

## Setup
Clone the repository
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For setting up torch with cuda you will have to add the following to ~/.bashrc
```
export CUDA_HOME=/usr/local/cuda/bin
export Path=/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

Note the dataset is already prepared and saved in Market/pytorch

## Question 1 & 2

No quatization with GPU (Solution for question 1)
```
python3 run_model.py --quantization None -–gpu
```
No quantization with CPU
```
python3 run_model.py --quantization None -–no-gpu
```
Static quantization to int 8 (Solution for question 2)
```
python3 run_model.py --quantization int8
```
Dynamic quantization to fp16
```
python3 run_model.py --quantization fp16
```

The extracted features are saved in output_mat/\

### Results:
int8 can be run in CPU and gives a ~1.6x speed up, ~3.5x decrease in model size with ~4% drop in MAP

---------- GPU-----------
>19732it [01:09, 282.02it/s]\
>3368it [00:13, 242.50it/s]\
>Training complete in 1m 23.86s\
>ft_net_dense\
>tee: ./model_int8/ft_net_dense/result.txt: No such file or directory\
>torch.Size([3368, 512])\
>Rank@1:0.901722 Rank@5:0.962292 Rank@10:0.974762 mAP:0.740243\

--------- CPU-----------
>19732it [21:03, 15.61it/s]\
>3368it [03:34, 15.72it/s]\
>Training complete in 24m 38.20s\
>ft_net_dense\
>torch.Size([3368, 512])\
>Rank@1:0.901722 Rank@5:0.962292 Rank@10:0.974762 mAP:0.740243\

-------- int8 static-----------
>Size of model before quantization\
>Size (MB): 34.625733\
>Size of model after quantization\
>Size (MB): 9.446883\
>19732it [13:33, 24.26it/s]\
>3368it [02:00, 27.93it/s]\
>Training complete in 15m 33.80s\
>ft_net_dense\
>torch.Size([3368, 512])\
>Rank@1:0.887173 Rank@5:0.957838 Rank@10:0.972090 mAP:0.712144 \

-------Tried fp16 dynamic -----------
>Size of model before quantization\
>Size (MB): 34.625733\
>Post Training Quantization: Convert done\
>Size of model after quantization\
>Size (MB): 34.617479\
>19732it [20:33, 15.99it/s]\
>3368it [03:27, 16.26it/s]\
>Training complete in 24m 0.81s\
>ft_net_dense\
>torch.Size([3368, 512])\
>Rank@1:0.901425 Rank@5:0.962292 Rank@10:0.974762 mAP:0.740222\

## Question 3
The answer is in question_3_answer.py\
The code used to test the code is in question_3_test.py if needed\

## References:
- https://github.com/layumi/Person_reID_baseline_pytorch.git\
- https://pytorch.org/docs/stable/quantization.html\
- Dataset - https://www.kaggle.com/datasets/sachinsarkar/market1501?resource=download\






