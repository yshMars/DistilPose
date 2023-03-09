# DistilPose:  Tokenized Pose Regression with Heatmap Distillation
Implementation for: DistilPose:  Tokenized Pose Regression with Heatmap Distillation

> [**DistilPose:  Tokenized Pose Regression with Heatmap Distillation**](https://arxiv.org/pdf/2303.02455.pdf),            
> Suhang Ye\*, Yingyi Zhang\*, Jie Hu\*, Liujuan Cao, Shengchuan Zhang<sup>✉</sup>, Lei Shen, Jun Wang, Shouhong Ding, Rongrong Ji. \
> In: Conference on Computer Vision and Pattern Recognition~(CVPR), 2023   
> *arXiv preprint ([arXiv 2303.02455](https://arxiv.org/abs/2303.02455))*  
> (\* equal contribution)

# Introdoction
In the field of human pose estimation, regression-based methods have been dominated in terms of speed, while heatmap-based methods are far ahead in terms of performance. How to take advantage of both schemes remains a challenging problem. In this paper, we propose a novel human pose estimation framework termed ***DistilPose***, which *bridges the gaps between heatmap-based and regression-based methods*. 

![image](https://github.com/yshMars/DistilPose/blob/main/images/framework.png)

Our contributions are summarized as follows:
- We propose a novel human pose estimation framework, ***DistilPose***, which is *the first work to transfer knowledge between heatmap-based and regression-based models losslessly*.
- We introduce a novel **Token-distilling Encoder (TDE)** to take advantage of both heatmap-based and regression-based models. With the proposed TDE, the gap between the output space of heatmaps and coordinate vectors can be facilitated in a tokenized manner.
- We propose **Simulated Heatmaps** to model explicit heatmap information, including 2D keypoint distributions and keypoint confidences. With the aid of Simulated Heatmaps, we can transform the regression-based HPE task into a more straightforward learning task that fully exploits local information. Simulated Heatmaps can be applied to any heatmap-based and regression-based models for transferring heatmap knowledge to regression models.

# Code
***Code will be published ASAP.***