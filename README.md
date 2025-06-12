# README

这是一个在 rk3588 上运行 yolo 检测的项目，支持 yolov5，yolov8 和 yolov11 的 [检测模型](https://docs.ultralytics.com/zh/tasks/detect/)，兼容海康威视 MVS USB 摄像头和 UVC 协议摄像头。

## 安装步骤

`letter_box`使用 pybind 实现，所以需要在 `src/letter_box_neon` 中编译`letter_box_neon.so`

```shell
cd src/letter_box_neon
mkdir build
cmake ..
make
cp letter_box_neon.so ../../..
```

运行安装脚本：

```bash
sudo ./install.sh
```

## 启动服务

```bash
sudo systemctl start camera-detector
sudo systemctl enable camera-detector
```

## 其他

模型更新删除启用等等操作需要在云平台上，也可以自行开发 mqtt 控制服务。

以下是模型的`config.json`文件格式：

```json
{
  "model_id": "2762e899-e084-45f0-ab56-dd2ed683b271",
  "sha256": "6e3c79dafa2ee732623a38cb1112877c31716242db6f00ad506c83521b11e60f",
  "name": "best.rknn",
  "model_type": "yolov8",
  "model_task": "detect",
  "version": "1",
  "classes": [
    "1",
    "2"
  ],
  "img_size": [
    640,
    640
  ]
}
```

如果是 yolov5, 还需要`anchors`字段。
