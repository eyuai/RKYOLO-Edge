# letterbox neon 版本

## 编译 `letter_box.cpp` 的指南

按照以下步骤编译 `letter_box.cpp` 文件：

### 前提条件

在开始之前，请确保您的系统已安装以下工具和库：

- **Python 3.12**
- **CMake**
- **g++**
- **pybind11**

### 编译步骤

1. **创建并激活虚拟环境**：

    ```sh
    python3.12 -m venv venv
    source venv/bin/activate
    ```

2. **安装 pybind11**：

    ```sh
    pip install pybind11
    ```

3. **创建构建目录**：

    ```sh
    mkdir build
    cd build
    ```

4. **运行 CMake 命令**：

    ```sh
    cmake -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir) ..
    ```

5. **编译项目**：

    ```sh
    make
    ```

6. **复制生成的共享库**：

    ```sh
    cp letterbox.so ..
    ```

7. **测试**:

    ```shell
    pip install numpy
    python test_letter_box_neon.py
    ```

完成这些步骤后，就能能够成功编译 `letter_box.cpp` 并生成共享库 `letterbox.so`。将该库复制到项目根目录以便使用。
