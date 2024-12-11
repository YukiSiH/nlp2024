README for Running PTQ-acc-cal on AutoDL

# 准备工作
选择镜像：选用包含PyTorch 1.7+和Python 3.6+的Ubuntu镜像。

# 配置实例
连接到实例：
使用SSH或AutoDL提供的JupyterLab连接实例。
设置Conda环境：
创建或激活Conda环境：
bash
conda env create --file env.yml
conda activate <environment_name>
若环境已存在，确保环境包含所有必要的包。
克隆项目：
在实例上克隆项目：
bash
git clone https://github.com/Guoxoug/PTQ-acc-cal.git
或者直接复制（我是这样做的）
配置路径（**必要步骤**）：
进入experiment_configs目录，编辑配置文件：
修改 "datapath" 指向数据集路径。
设置 "results_savedir" 为结果保存路径。

# 运行项目
导航到脚本目录：
进入 experiment_scripts 目录：
bash
cd ~/PTQ-acc-cal-main/experiment_scripts
使脚本可执行：
运行：
bash
chmod +x *
运行训练和测试脚本：
例如：
bash
./<model>_<dataset>_train.sh
./<model>_<dataset>_test.sh
替换 <model> 和 <dataset> 为具体配置。
监控和保存结果：
通过SSH或JupyterLab监控进程。
结果会保存到你配置的路径。

# 注意事项
数据集：确保数据集（如CIFAR-100或ImageNet）上传或可访问。
存储：为项目结果分配足够的存储空间。