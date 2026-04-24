# Streamlit System Prototype

本目录包含毕设“系统设计与实现”阶段使用的 Streamlit 原型系统。

## 入口

```bash
streamlit run app/streamlit_app.py
```

## 页面内容

- 首页概览：系统定位、模型目录、关键指标
- 实时推理：支持单张图片上传与双模型对比
- 批量推理：支持多张图片推理与 CSV 导出
- 实验看板：展示论文曲线、表格与精度-复杂度关系
- 系统设计：展示系统模块、处理流程与代码结构

## 设计风格

页面视觉语言参考 `VoltAgent / awesome-design-md`：

- 深色背景
- 祖母绿高亮
- 技术仪表盘与终端感排版

参考地址：

```text
https://github.com/VoltAgent/awesome-design-md
```

## 代码结构

- `streamlit_app.py`：页面入口与界面布局
- `system_runtime.py`：模型加载、预处理、推理与结果整理
