# MEMO

clip = GILL

## TODO

- 加师弟git权限 分支代码一定要写在trial分支上
  - 分支clone: git clone git@github.com:ZacBi/SNPA.git
    - 如果clone失败， 参考[配置ssh连接git](https://www.cnblogs.com/OnlyAR/p/16155406.html)
  - 切换到分支上: git checkout trial
- eva的参考资料和模型文件给师弟
  - follow eva的[setup](https://github.com/baaivision/EVA/blob/master/EVA-02/det/README.md)
  - eva的[modelscope模型文件](https://www.modelscope.cn/models/zacbi2023/eva02/summary)
- coco2017 的数据集都给师弟
  - [数据集地址](https://www.modelscope.cn/datasets/zacbi2023/coco2017_caption/summary)
- modelscope使用方式: [modelscope doc](https://www.modelscope.cn/docs/%E9%A6%96%E9%A1%B5)
- poetry文件打包好，让师弟安装好（废弃）

## 5.1

- clip + eva跑通
  - online：模型架构拼接在一起，跑通
  - offline: 中间结果要生成，sd的结果和eva的结果能对应
- 图像切割的结果的结构识别，即如何判断某个像素点属于哪一个cls
- 桃陈：怎么算输入对于输出的贡献(画热力图)，得参考之前的那篇文章的代码(人骑马)
- 师弟：把同时符合gill和eva的环境安装好
- 俊峰: intro写一半, 逻辑整理好

## 5.2 | 5.3

- 师弟去跑激活值的钩子函数
- llm的输出和sd输出之间的关系
- 看clip怎么加hook + 桃陈来跑导数的钩子函数

## 5.4 | 5.5

- 神经元获取和保存
- 师弟看GILL里面文字（输入） -> 图片/文字（输出）的神经元重叠情况