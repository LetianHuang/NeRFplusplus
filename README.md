# NeRF++

## Introduction

1. This project contains a reproduction of the [NeRF++](https://arxiv.org/abs/2010.07492) paper. For reference only!
2. Using the PyTorch framework, V100 single card GPU. The specific environment configuration is as follows.
3. The reasons for the success of NeRF are analyzed and the methods used to solve the problem of 360 degree scene reconstruction are proposed. The main method is the technique of separating foreground and background, the original NeRF is used for foreground, and the inverse spherical parameterization is used for background. 

## Environment and configuration

* OS Operating System: Linux Ubuntu 18.04.4 LTS (GNU/Linux 4.15.0-180-generic x86_64)
* GPU graphics card resources: Tesla V100-SXM2-32GB
* CUDA version: 11.4
* Python version: 3.8.12
* torch version: 1.9.0+cu102
* torchvision version: 0.10.0+cu102
* cudnn version: 8.2.4

Other python module dependencies

```
tqdm
opencv-python
scikit-image==0.19.3
```

## 3D scene visualization

<table>
<tr>
    <th> Scene </th> <th>Ground Truth</th> <th>Predict</th>
</tr>

<tr>
<td> Truck </td> <td width=50%><img src="./GTTruck.gif"></td><td width=50%><img src="./TestTruck.gif"> </td>
</tr>
</table>

Table 1: Use NeRF++ to reconstruct the 360 degree scene and compare the Ground Truth with the prediction

## Experiment

| Model | PSNR $\uparrow$ | SSIM $\uparrow$ |
|  --- | --- | --- | 
| NeRF(in paper) | 20.85 | 0.747 |
| NeRF++(The project) | 22.09 | 0.801 |
| NeRF++(in paper) | **22.77** | **0.823** |

Table 2: The results in this project are compared with the original paper. We find that the performance of NeRF++ implemented in this project is greatly improved compared with NeRF, but there is still a little gap compared with the implementation in the paper, which should be due to the author's partial trick that I ignored when I reproduced it. 

## References

Kai Zhang, Gernot Riegler, Noah Snavely, Vladlen Koltun. NERF++: ANALYZING AND IMPROVING NEURAL RADIANCE FIELDS. In [https://arxiv.org/abs/2010.07492](https://arxiv.org/abs/2010.07492). 
