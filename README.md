[colab地址](https://colab.research.google.com/drive/1dd0WT9BmFmVWGnxFYNcKNjbeJHwYlHEo?usp=sharing)

[原readme文档](https://github.com/thuml/Transfer-Learning-Library)

## 说明

本仓库代码用自己的数据集训练迁移学习模型，基于清华迁移学习库dalib修改

## Q&A

### 如何分析DANN方法初始数据的分布？

   答：TSNE图，源域是红色，目标域是蓝色

  注释掉 /content/Transfer-Learning-Library/examples/domain_adaptation/classification/damm_wad.py以下几行：

  ```
  # resume from the best checkpoint
  if args.phase != 'train':
      checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
      classifier.load_state_dict(checkpoint)
  ```
  
  然后直接执行：
  
  ```
  # DANN分析自己数据集
  %cd /content/Transfer-Learning-Library/examples/domain_adaptation/classification
  !CUDA_VISIBLE_DEVICES=0 python dann_wad.py data/MRSSC -d MRSSC -s V -t I -v T -a resnet50 --epochs 10 --seed 1 --log logs/dann/MRSSC_V2I --phase analysis
  ```

### 怎么输出混淆矩阵？

   答：命令行加参数 '--per-class-eval'，去代码搜索 ConfusionMatrix 就能看到相应代码，修改代码可以print更多自己想要的
