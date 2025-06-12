#!/bin/bash
# 1. 检查nginx libnginx-mod-rtmp libglu1-mesa-dev  libglu1-mesa python3.12 python3.12-venv redis 是否安装，未安装则提醒是否安装
# 2. 创建.venv 目录，并安装依赖
# 3. 检查 rknn-toolkit-lite2 是否安装，未安装则下载并安装
# 4. 执行generate_cert.sh生成签名证书，并配置开机运行generate_cert.sh 配置签名证书
# 5. 创建/etc/camera.d目录，并复制config.py到该目录
# 6. 创建/etc/systemd/system/camera-detector.service文件，并配置服务
# 7. 启动服务
# 8. 拷贝camera.conf到/etc/nginx/conf.d目录
# 9. 修改 nginx user 为 root ，重启 nginx

set -eo pipefail

# 颜色变量
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

trap 'echo -e "${RED}脚本执行中遇到错误，已退出。${NC}"' ERR

SERVICE_NAME="camera-detector"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME.service"
CONFIG_DIR="/etc/camera.d"
CONFIG_FILE="$CONFIG_DIR/config.py"
PROJECT_PATH=$(realpath "$(pwd)")
SOFTWARE_DIR="$HOME/software"
DELETE_TEMP_FILES=false
FORCE_ENV=false


# 获取当前用户信息
function get_user_info {
    USERNAME=$(whoami)
    USER_HOME="$HOME"
}

# 检查服务是否已安装
function check_service {
    if systemctl list-units --type=service --all | grep -q "$SERVICE_NAME.service"; then
        echo -e "${YELLOW}检测到 $SERVICE_NAME 已安装，是否重新安装？(y/N) ${NC}"
        read -r choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}正在卸载旧版本...${NC}"
            remove_old_installation
        else
            echo -e "${GREEN}已取消安装。${NC}"
            exit 0
        fi
    fi
}

# 移除旧的安装（保留配置文件）
function remove_old_installation {
    sudo bash -c "
        systemctl stop \"$SERVICE_NAME\" 2>/dev/null || true
        systemctl disable \"$SERVICE_NAME\" 2>/dev/null || true
        rm -f \"$SERVICE_PATH\"
        systemctl daemon-reload
    "

    echo -e "${GREEN}旧版本已移除。${NC}"
}

# 检查必要的软件包是否存在
function check_required_packages {
    echo -e "${BLUE}检查必要的软件包是否存在...${NC}"
    # 检查 nginx
    if ! command -v nginx &>/dev/null; then
        echo -e "${YELLOW}缺少 nginx，请手动安装！${NC}"
        read -p "是否安装 nginx？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y nginx
        else
            echo -e "${RED}请手动安装 nginx！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}nginx 已存在，跳过检查...${NC}"
    fi
    # 检查 libnginx-mod-rtmp
    if ! dpkg -l | grep -E '^ii\s+libnginx-mod-rtmp'; then
        echo -e "${YELLOW}缺少 libnginx-mod-rtmp，请手动安装！${NC}"
        read -p "是否安装 libnginx-mod-rtmp？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y libnginx-mod-rtmp
        else
            echo -e "${RED}请手动安装 libnginx-mod-rtmp！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}libnginx-mod-rtmp 已存在，跳过检查...${NC}"
    fi
    # 检查 libglu1-mesa-dev
    if ! dpkg -l | grep -E '^ii\s+libglu1-mesa-dev'; then
        echo -e "${YELLOW}缺少 libglu1-mesa-dev，请手动安装！${NC}"
        read -p "是否安装 libglu1-mesa-dev？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y libglu1-mesa-dev
        else
            echo -e "${RED}请手动安装 libglu1-mesa-dev！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}libglu1-mesa-dev 已存在，跳过检查...${NC}"
    fi

	# 检查 libavutil-dev
	if ! dpkg -l | grep -E '^ii\s+libavutil-dev'; then
        echo -e "${YELLOW}缺少 libavutil-dev，请手动安装！${NC}"
        read -p "是否安装 libavutil-dev？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y libavutil-dev
        else
            echo -e "${RED}请手动安装 libavutil-dev！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}libavutil-dev 已存在，跳过检查...${NC}"
    fi

    # 检查 libglu1-mesa
    if ! dpkg -l | grep -E '^ii\s+libglu1-mesa'; then
        echo -e "${YELLOW}缺少 libglu1-mesa，请手动安装！${NC}"
        read -p "是否安装 libglu1-mesa？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y libglu1-mesa
        else
            echo -e "${RED}请手动安装 libglu1-mesa！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}libglu1-mesa 已存在，跳过检查...${NC}"
    fi
    # 检查 python3.12
    if ! command -v python3.12 &>/dev/null; then
        echo -e "${YELLOW}缺少 python3.12，请手动安装！${NC}"
        read -p "是否安装 python3.12？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository ppa:deadsnakes/ppa -y
            sudo apt-get update
            sudo apt-get install -y python3.12 python3.12-venv
        else
            echo -e "${RED}请手动安装 python3.12！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}python3.12 已存在，跳过检查...${NC}"
    fi
    # 检查 redis
    if ! command -v redis-server &>/dev/null; then
        echo -e "${YELLOW}缺少 redis，请手动安装！${NC}"
        read -p "是否安装 redis？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y redis-server
        else
            echo -e "${RED}请手动安装 redis！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}redis 已存在，跳过检查...${NC}"
    fi
    # 检查 mkcert
    if ! command -v mkcert &>/dev/null; then
        echo -e "${YELLOW}缺少 mkcert，请手动安装！${NC}"
        read -p "是否安装 mkcert？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y mkcert
        else
            echo -e "${RED}请手动安装 mkcert！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}mkcert 已存在，跳过检查...${NC}"
    fi

	if ! command -v v4l2-ctl &>/dev/null; then
        echo -e "${YELLOW}缺少 v4l2-ctl（v4l-utils），请手动安装！${NC}"
        read -p "是否安装 v4l-utils？(y/N) " choice
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y v4l-utils
        else
            echo -e "${RED}请手动安装 v4l-utils！${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}v4l2-ctl 已存在，跳过检查...${NC}"
    fi
}

# 安装 rknn-toolkit-lite2
function install_rknn_toolkit {
    echo -e "${BLUE}检测当前 Python 版本和架构...${NC}"

    # 确保 python3.12 存在
    if ! command -v python3.12 &>/dev/null; then
        echo -e "${RED}未检测到 Python 3.12，请先安装 Python 3.12！${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python3.12 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    ARCH=$(uname -m)

    echo "当前 Python 版本: $PYTHON_VERSION"
    echo "当前架构: $ARCH"

    # 生成正确的 whl 文件名
    WHEEL_FILE="rknn_toolkit_lite2-2.3.0-${PYTHON_VERSION}-${PYTHON_VERSION}-manylinux_2_17_${ARCH}.manylinux2014_${ARCH}.whl"
    # WHEEL_URL="${GITHUB_MIRROR}https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit-lite2/packages/$WHEEL_FILE"
    WHEEL_URL="https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit-lite2/packages/$WHEEL_FILE"
    WHEEL_PATH="$SOFTWARE_DIR/$WHEEL_FILE"

    # 确保 ~/software 目录存在
    mkdir -p "$SOFTWARE_DIR"

    # 检查文件是否存在
    if [ -f "$WHEEL_PATH" ]; then
        echo -e "${YELLOW}检测到 $WHEEL_PATH 已存在，尝试安装...${NC}"
        if ! pip install "$WHEEL_PATH"; then
            echo -e "${RED}检测到 $WHEEL_PATH 无效，删除并重新下载...${NC}"
            rm -f "$WHEEL_PATH"
        fi
    fi

    # 如果文件不存在或无效，重新下载
    if [ ! -f "$WHEEL_PATH" ]; then
        echo -e "${BLUE}下载 $WHEEL_URL 到 $WHEEL_PATH ...${NC}"
        if ! wget -O "$WHEEL_PATH" -c "$WHEEL_URL"; then
            echo -e "${RED}下载失败，请检查网络或手动下载 ${WHEEL_URL}！${NC}"
            exit 1
        fi
    fi

    echo -e "${BLUE}安装 $WHEEL_PATH ...${NC}"
    if ! pip install "$WHEEL_PATH"; then
        echo -e "${RED}安装失败！${NC}"
        exit 1
    fi

    if [ "$DELETE_TEMP_FILES" = true ]; then
        echo -e "${GREEN}删除临时文件 $WHEEL_PATH${NC}"
        rm -f "$WHEEL_PATH"
    fi
}

# 生成签名证书
function generate_certificates {
    echo -e "${BLUE}生成签名证书...${NC}"
    if [ -f "./generate_cert.sh" ]; then
        bash ./generate_cert.sh
    else
        echo -e "${RED}generate_cert.sh 文件不存在！请确保文件在当前目录下。${NC}"
        exit 1
    fi
}

# 配置开机运行 generate_cert.sh
function configure_generate_cert_service {
    echo -e "${BLUE}配置开机运行 generate_cert.sh...${NC}"
    MKCERT_SERVICE_PATH="/etc/systemd/system/generate-cert.service"

    cat <<EOF | sudo tee "$MKCERT_SERVICE_PATH" > /dev/null
[Unit]
Description=Generate Certificates using mkcert
After=network.target

[Service]
Type=oneshot
ExecStart=mkcert -install
ExecStart=bash $PROJECT_PATH/generate_cert.sh
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable generate-cert
    sudo systemctl start generate-cert

    echo -e "${GREEN}generate_cert.sh 已配置为开机运行。${NC}"
}

# 创建 /etc/camera.d 目录，并复制 config.py 到该目录
function copy_config_file {
    echo -e "${BLUE}检查配置文件... ${NC}"
    sudo mkdir -p "$CONFIG_DIR"
    if [ "$FORCE_ENV" = true ] || [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${BLUE}正在复制默认配置文件到 $CONFIG_FILE ...${NC}"
        sudo cp config.py "$CONFIG_FILE"
        sudo chmod 644 "$CONFIG_FILE"
    else
        echo -e "${YELLOW}检测到 $CONFIG_FILE 已存在，跳过复制...${NC}"
    fi
}

# 配置 systemd 服务
function configure_systemd_service {
    echo -e "${BLUE}正在配置服务文件... ${NC}"
    SERVICE_FILE="/tmp/camera-detector.tmp"

    sed "s|__PROJECT_PATH__|$PROJECT_PATH|g; s|__USERNAME__|$USERNAME|g" camera-detector.service > "$SERVICE_FILE"

    # 安装 systemd 服务
    sudo bash -c "
        if ! cp \"$SERVICE_FILE\" \"$SERVICE_PATH\"; then
            echo -e \"${RED}复制服务文件失败！${NC}\"
            exit 1
        fi
        rm -f \"$SERVICE_FILE\"

        systemctl daemon-reload
        systemctl enable \"$SERVICE_NAME\"
        systemctl restart \"$SERVICE_NAME\"
    "

    echo -e "${GREEN}服务已安装到：${SERVICE_PATH}"
    echo -e "${GREEN}如需查看日志，请执行：journalctl -u $SERVICE_NAME -f${NC}"
}

# 拷贝 camera.conf 到 /etc/nginx/conf.d 目录
function copy_camera_conf {
    echo -e "${BLUE}拷贝 camera.conf 到 /etc/nginx/conf.d 目录... ${NC}"
    if [ -f "./camera.conf" ]; then
        sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak
        sudo cp nginx.conf /etc/nginx/nginx.conf
        sudo cp camera.conf /etc/nginx/conf.d/
        sudo nginx -t
        if [ $? -ne 0 ]; then
            echo -e "${RED}Nginx 配置文件语法错误，请检查 camera.conf！${NC}"
            exit 1
        fi
		sudo systemctl restart nginx
		if [ $? -ne 0 ]; then
            echo -e "${RED}重启 Nginx 失败，请检查 Nginx 配置！${NC}"
            exit 1
        fi
        echo -e "${GREEN}Nginx 用户已修改为 root 并重启成功。${NC}"
    else
        echo -e "${RED}camera.conf 文件不存在！请确保文件在当前目录下。${NC}"
        exit 1
    fi
}


# 处理参数
function parse_arguments {
    while getopts ":cf" opt; do
        case ${opt} in
            c )
                DELETE_TEMP_FILES=true
                ;;
            f )
                FORCE_ENV=true
                ;;
            \? )
                echo -e "${RED}无效的选项: -$OPTARG${NC}" 1>&2
                exit 1
                ;;
        esac
    done
    shift $((OPTIND -1))
}

# 初始化虚拟环境
function init_virtualenv {
    if [ "$FORCE_ENV" = true ]; then
        echo -e "${BLUE}强制使用新的虚拟环境... ${NC}"
        if [ -d "$VENV_PATH" ]; then
            echo -e "${YELLOW}删除现有虚拟环境 $VENV_PATH ...${NC}"
            rm -rf "$VENV_PATH"
        fi
        python3.12 -m venv "$VENV_PATH"
    else
        echo -e "${BLUE}正在初始化虚拟环境... ${NC}"
        if [ ! -d "$VENV_PATH" ]; then
            python3.12 -m venv "$VENV_PATH"
        else
            echo -e "${YELLOW}虚拟环境已存在，跳过初始化...${NC}"
        fi
    fi
    source "$VENV_PATH/bin/activate"
}

# 安装 requirements.txt 依赖
function install_requirements {
    echo -e "${BLUE}安装 requirements.txt 依赖...${NC}"
    if ! pip install -r requirements.txt; then
        echo -e "${RED}依赖安装失败，请检查 requirements.txt！${NC}"
        exit 1
    fi
}

# 主函数
function main {
    echo -e "${BLUE}正在初始化配置... ${NC}"

    # 解析参数
    parse_arguments "$@"

    # 获取用户信息
    get_user_info

    # 检查必要的软件包是否存在
    check_required_packages

    # 修正 VENV_PATH 赋值
    VENV_PATH="$USER_HOME/.venv"

    # 检查是否已安装
    check_service

    # 初始化虚拟环境
    init_virtualenv

    # 安装 requirements.txt 依赖
    install_requirements

    # 安装 rknn-toolkit-lite2
    install_rknn_toolkit

    # 生成签名证书
    generate_certificates

    # 配置开机运行 generate_cert.sh
    configure_generate_cert_service

    # 复制配置文件（如果不存在或使用 -f 参数）
    copy_config_file

    # 配置 systemd 服务
    configure_systemd_service

    # 拷贝 camera.conf 到 /etc/nginx/conf.d 目录
	# 拷贝 nginx.conf 到 /etc/nginx 目录
    copy_camera_conf


    echo -e "${GREEN}安装完成！"
}

# 启动主函数
main "$@"