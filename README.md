# DINOv2-Segmentation
  环境配置根据DINOv2 - https://github.com/facebookresearch/dinov2
# 下载模型参数
  此文件中创建 model文件夹放入，25epoch是精度最高的模型，100epoch是最后训练loss最低的模型
# 训练代码
  完整训练在 Train.ipynb 中，可以直接打开观看部分可视化结果
# 训练Loss可视化
  需要安装 TensorBoard 
  ```pip install rensorboard ```
  ``` tensorboard --logdir=runs/segmentation ```
# 测试代码
  测试代码在 predict.ipynb 中
  只需要更改
  ```checkpoint_path = 'model/xxx.pth'```
``` img_path = 'dataset/image/cityscapes_xxx/xxx.png'```
```ground_truth_path = /dataset/label/cityscapes_19classes_xxx/xxx.png‘```
  ! 注意test数据集的 ground_truth 是完全空白的，但是路径也需要写入代码
