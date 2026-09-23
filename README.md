# 谷粒 拍照 秒出 谷粒数（Grain Photo Counter）

安卓 App：拍一张稻谷/米粒散粒照片，秒出谷粒数 + 逐粒标注图。

## 功能

- 📷 **拍照计数**：直接调系统相机拍稻谷照片
- 🖼 **相册选图计数**：从相册选一张散粒照片
- ⚡ **秒出结果**：计数 + 标注图（绿框=独立单粒+红色编号，橙框=粘连团按面积折算"N粒"）

## 原代码出处（重要）

核心计数算法来自开源项目：

> **[stars-spark/asw-spc-rice-counting](https://github.com/stars-spark/asw-spc-rice-counting)**
> ASW-SPC（Adaptive-Scale marker Watershed with Shape-Prior Correction，尺度自标定 + 标记控制注水分割 + 形状先验校正）
> 作者：stars-spark · MIT License · 数字图像处理课程作业 · 2026-09

本仓库把该算法的核心模块（`counter` / `preprocess` / `calibrate` / `segment` / `correct`）重写为**纯 numpy + opencv** 实现，去除了 skimage/scipy 依赖（便于安卓打包），重写版与原版计数结果完全一致（实测同一张稻谷图均得 115 粒）。

算法文件：`app/src/main/python/asw_core.py`（文件内注释标明了各函数与原项目的对应关系）。

## 技术栈

| 项 | 说明 |
|----|------|
| 打包 | Chaquopy 17.0（Python 嵌入 Android） |
| Python | 3.10（opencv 预编译只到 cp310） |
| 依赖 | numpy 1.26 + opencv-python-headless 4.5 |
| 算法 | ASW-SPC 纯 numpy+opencv 重写版 |
| minSdk | 24（Android 7.0+），arm64-v8a |

## 构建

环境要求：JDK 17 + Android SDK（compileSdk 34）+ Gradle 8.5 + 本机 Python 3.10。

```bash
# 设置 JAVA_HOME / ANDROID_HOME 后
gradle assembleDebug --no-daemon
# APK 输出在 app/build/outputs/apk/debug/app-debug.apk
```

## 下载

预编译 APK 在 [Releases](../../releases) 页面下载。

## 目录结构

```
├─ settings.gradle            # 仓库 + 插件仓库配置
├─ build.gradle               # AGP + Chaquopy 插件版本
├─ gradle.properties
└─ app/
   ├─ build.gradle            # Chaquopy 配置（Python 3.10 + pip 依赖）
   └─ src/main/
      ├─ AndroidManifest.xml  # PyApplication + FileProvider
      ├─ java/.../MainActivity.java   # 拍照/选图 + 显示结果
      ├─ python/
      │  ├─ main.py           # count_and_label 入口
      │  └─ asw_core.py       # ASW-SPC 纯 numpy+opencv 重写
      └─ res/xml/file_paths.xml
```

## License

MIT License（继承自原项目 stars-spark/asw-spc-rice-counting 的 MIT 协议）。
