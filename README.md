# STNet + LPRNet
STNet 进行空间校正 + LPRNet 进行车牌识别

目前在 CCPD 数据集下，各类作为一个整体，以 half 模式运行测试，最终 macc 约为 95.1%

## 使用
1. 运行下列命令进行训练

    ```sh
    python train.py --source-dir /data/CCPD2019
    ```
   
2. 运行下列命令进行测试

    ```sh
    python test.py --source-dir /data/CCPD2019 --weights weights/final.pt
    ```

3. ONNX

    ```sh
    python export.py --weights weights/final.pt
    ```

## 目录结构
```
/data 数据加载器
|-- ccpd.py 用于遍历 source-dir 目录中的 CCPD 数据集，提供基本的数据解析
|-- bases.py 数据集基类操作
|-- dataset.py 基于 CCPD 数据集制作的 Pytorch Dataset 实现
/model 模型
|-- lprnet.py LPRNet 实现
|-- st.py STNet 实现
/runs
|-- /cache 缓存文件
|-- /exp* 训练时的数据输出目录，每次启动会自动自增 + 1
    |-- weights 输出的模型文件
        |-- last.pt 目前最新一次完成的 epoch 保存的数据
        |-- best.pt 训练过程中，测试得到最高 acc 的模型
        |-- final.pt 跑完所有 epoch 后保存的最终模型文件，注意，不应使用此文件继续训练，请使用 last.pt 继续训练
/weights
|-- final.pt 已训练好的模型，方便快速开始测试、产品部署，但不应用其继续训练
```

## Links
* [原仓库](https://github.com/Cat7373/STNet_LPRNet/blob/master/data/ccpd.py)
    * 模型和原始代码来自这里

## TODO
* [ ] 双行车牌支持
