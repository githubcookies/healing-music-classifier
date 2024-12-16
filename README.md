# 治愈音乐分类器 (Healing Music Classifier)

![项目Logo](https://img.shields.io/badge/AI-Music%20Classifier-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 项目简介

治愈音乐分类器是一个基于机器学习的音乐分析工具，能够自动评估音乐的治愈潜力。该项目使用先进的音频特征提取技术和随机森林算法，对音乐的治愈属性进行量化分析。

### 在线使用

你可以直接通过以下链接使用我们的在线演示版本：
[Healing Music Classifier App](https://huggingface.co/spaces/404Brain-Not-Found-yeah/healing-music-classifier-app)

## 功能特点

- 🎵 支持多种音频格式（MP3、WAV）
- 🎼 专业的音频特征提取（MFCC、频谱特征等）
- 🤖 基于随机森林的机器学习分类
- 📊 直观的治愈指数可视化
- 🌐 便捷的Web界面
- ☁️ 支持在线部署

## 技术实现

### 音频特征提取
项目使用了以下关键特征：
- MFCC（梅尔频率倒谱系数）- 13个系数
- 色度特征（Chroma Features）- 12个特征
- 统计特征（均值和方差）

### 模型架构
- 使用随机森林分类器
- 特征标准化处理
- 交叉验证确保模型稳定性

## 本地安装使用

### 环境要求
- Python 3.8+
- pip包管理器

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/githubcookies/healing-music-classifier.git
cd healing-music-classifier
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
- 创建`healing_music`文件夹存放治愈音乐样本
- 创建`non_healing_music`文件夹存放非治愈音乐样本

4. 训练模型
```bash
python train_model.py
```

5. 启动Web应用
```bash
streamlit run app.py
```

## 模型训练流程

1. 数据准备
   - 收集治愈音乐和非治愈音乐样本
   - 将音频文件分别放入对应文件夹

2. 特征提取
   - 使用librosa库提取音频特征
   - 包括MFCC、色度特征等

3. 模型训练
   - 数据预处理和标准化
   - 使用随机森林算法训练
   - 进行交叉验证评估

4. 模型保存
   - 保存训练好的模型
   - 保存特征缩放器

## 在线使用指南

1. 访问[在线演示](https://huggingface.co/spaces/404Brain-Not-Found-yeah/healing-music-classifier-app)
2. 点击"Choose an audio file..."上传音乐文件
3. 等待分析完成
4. 查看治愈指数和分析结果

### 分析结果说明
- 治愈指数 >= 75%: 强治愈效果
- 治愈指数 50-75%: 中等治愈效果
- 治愈指数 < 50%: 治愈效果有限

## 项目结构

```
healing-music-classifier/
├── app.py              # Streamlit Web应用
├── predict.py          # 预测功能模块
├── train_model.py      # 模型训练模块
├── requirements.txt    # 项目依赖
├── models/            # 模型文件夹
│   ├── model.joblib   # 训练好的模型
│   └── scaler.joblib  # 特征缩放器
├── healing_music/     # 治愈音乐样本
└── non_healing_music/ # 非治愈音乐样本
```

## 开发计划

- [ ] 添加更多音频特征支持
- [ ] 优化模型性能
- [ ] 添加批量处理功能
- [ ] 支持更多音频格式

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

该项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目链接：[https://github.com/githubcookies/healing-music-classifier](https://github.com/githubcookies/healing-music-classifier)
- HuggingFace空间：[https://huggingface.co/spaces/404Brain-Not-Found-yeah/healing-music-classifier-app](https://huggingface.co/spaces/404Brain-Not-Found-yeah/healing-music-classifier-app)

## 致谢

感谢改变世界的力课程团队，在课程大作业的push下，我们完成了这个项目。
特别感谢以下开源项目：
- librosa
- streamlit
- scikit-learn
- HuggingFace Spaces
