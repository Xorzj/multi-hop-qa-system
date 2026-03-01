#!/bin/bash
# ============================================================
#  AutoDL 快速部署脚本 — 多跳推理知识图谱问答系统
# ============================================================
#
#  使用方法:
#    1. 在 AutoDL 上租一台 GPU 实例 (推荐: RTX 3090/4090, 镜像选 PyTorch 2.x)
#    2. 把整个项目上传到 /root/autodl-tmp/PROJECT/
#       - 方法A: 先在本机 git push, 再在 AutoDL git clone
#       - 方法B: 用 AutoDL 的文件上传功能 (scp / rsync)
#    3. 运行: bash /root/autodl-tmp/PROJECT/scripts/setup_autodl.sh
#    4. 跑流水线: cd /root/autodl-tmp/PROJECT && uv run python run_pipeline.py
#
#  预计耗时: ~10分钟 (取决于网速)
#  磁盘需求: 约 20GB (模型 15GB + 依赖 + 项目)
# ============================================================

set -e # 任何命令失败立即退出

# ── 颜色输出 ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() {
	echo -e "${RED}[ERROR]${NC} $1"
	exit 1
}

# ── 配置 ─────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/PROJECT}"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR="${PROJECT_DIR}/models/qwen2.5-7b-instruct"
NEO4J_PASSWORD="password123"

# ── 0. 检查环境 ──────────────────────────────────────────────
info "检查环境..."
if ! command -v nvidia-smi &>/dev/null; then
	error "未检测到 GPU, AutoDL 实例必须带 GPU"
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
info "GPU: ${GPU_INFO}"

if [ ! -d "${PROJECT_DIR}" ]; then
	error "项目目录不存在: ${PROJECT_DIR}\n  请先上传项目到该路径"
fi

# ── 1. 安装 uv ───────────────────────────────────────────────
info "[1/6] 安装 uv 包管理器..."
if command -v uv &>/dev/null; then
	info "uv 已安装: $(uv --version)"
else
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
	# 写入 bashrc 以便后续使用
	echo 'export PATH="$HOME/.local/bin:$PATH"' >>~/.bashrc
	info "uv 安装完成: $(uv --version)"
fi

# ── 2. 安装 Python 依赖 ──────────────────────────────────────
info "[2/6] 安装 Python 依赖..."
cd "${PROJECT_DIR}"
uv sync --group dev --group dl --group llm --group graph --group api
info "依赖安装完成"

# ── 3. 下载 7B 模型 ──────────────────────────────────────────
info "[3/6] 下载模型: ${MODEL_NAME}..."
if [ -f "${MODEL_DIR}/config.json" ]; then
	info "模型已存在, 跳过下载"
else
	mkdir -p "${MODEL_DIR}"

	# 使用 hf-mirror.com 加速 (国内镜像)
	export HF_ENDPOINT="https://hf-mirror.com"
	info "使用镜像: ${HF_ENDPOINT}"

	uv run python -c "
from huggingface_hub import snapshot_download
print('开始下载 ${MODEL_NAME} ...')
path = snapshot_download(
    repo_id='${MODEL_NAME}',
    local_dir='${MODEL_DIR}',
    resume_download=True,
)
print(f'下载完成: {path}')
"
	info "模型下载完成"
fi

# ── 4. 更新配置为 7B 模型 ────────────────────────────────────
info "[4/6] 更新配置文件..."
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
if [ -f "${CONFIG_FILE}" ]; then
	# 将 model_path 替换为 7B 路径
	sed -i 's|model_path:.*|model_path: "models/qwen2.5-7b-instruct"|' "${CONFIG_FILE}"
	info "config.yaml 已更新: model_path → models/qwen2.5-7b-instruct"
else
	warn "config.yaml 未找到, 请手动配置"
fi

# ── 5. 启动 Neo4j ────────────────────────────────────────────
info "[5/6] 启动 Neo4j..."
export NEO4J_PASSWORD="${NEO4J_PASSWORD}"

if command -v docker &>/dev/null; then
	# 检查容器是否已在运行
	if docker ps --format '{{.Names}}' | grep -q multihop-qa-neo4j; then
		info "Neo4j 容器已在运行"
	else
		cd "${PROJECT_DIR}"
		docker compose up -d
		info "等待 Neo4j 启动..."
		sleep 10
		# 等待健康检查通过
		for i in $(seq 1 30); do
			if docker exec multihop-qa-neo4j cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" "RETURN 1" &>/dev/null; then
				info "Neo4j 已就绪"
				break
			fi
			if [ "$i" -eq 30 ]; then
				warn "Neo4j 启动超时, 请手动检查: docker logs multihop-qa-neo4j"
			fi
			sleep 2
		done
	fi
else
	warn "未安装 Docker, 请手动安装并启动 Neo4j:"
	warn "  apt-get update && apt-get install -y docker.io docker-compose-plugin"
	warn "  cd ${PROJECT_DIR} && docker compose up -d"
fi

# ── 6. 验证 ──────────────────────────────────────────────────
info "[6/6] 验证环境..."
cd "${PROJECT_DIR}"

echo ""
info "━━━ GPU ━━━"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
info "━━━ Python ━━━"
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
info "━━━ 模型 ━━━"
if [ -f "${MODEL_DIR}/config.json" ]; then
	MODEL_SIZE=$(du -sh "${MODEL_DIR}" | cut -f1)
	info "模型路径: ${MODEL_DIR} (${MODEL_SIZE})"
else
	warn "模型未下载完成!"
fi

echo ""
info "━━━ Neo4j ━━━"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q multihop-qa-neo4j; then
	info "Neo4j 运行中 → http://localhost:7474"
else
	warn "Neo4j 未运行"
fi

# ── 完成 ─────────────────────────────────────────────────────
echo ""
echo "=========================================="
info "部署完成!"
echo "=========================================="
echo ""
echo "  下一步:"
echo "    cd ${PROJECT_DIR}"
echo "    export NEO4J_PASSWORD=${NEO4J_PASSWORD}"
echo "    uv run python run_pipeline.py"
echo ""
echo "  其他命令:"
echo "    uv run pytest                          # 跑测试"
echo "    uv run uvicorn api.main:app --port 8000  # 启动API"
echo "    uv run ruff check src/ api/            # 代码检查"
echo ""
