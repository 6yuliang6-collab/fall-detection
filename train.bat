@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  跌倒检测模型训练脚本（双击运行，改下面参数即可）
REM ============================================================
set IMG=640
set BATCH=16
set EPOCHS=150
set DATA=data.yaml
set WEIGHTS=yolov5s.pt
set DEVICE=0
set WORKERS=4
set HYP=yolov5\data\hyps\hyp.fall.yaml
set PATIENCE=30

REM ============================================================
REM  自动定位到脚本所在目录（项目根目录），无需手动 cd
REM ============================================================
cd /d "%~dp0"

REM ---------- 检查关键文件是否存在 ----------
if not exist "venv\Scripts\python.exe" (
    echo [错误] 找不到虚拟环境 venv\Scripts\python.exe
    pause
    exit /b 1
)
if not exist "yolov5\train.py" (
    echo [错误] 找不到 yolov5\train.py
    pause
    exit /b 1
)
if not exist "%DATA%" (
    echo [错误] 找不到数据集配置 %DATA%
    echo        请确认 dataset 目录和数据已准备好
    pause
    exit /b 1
)
if not exist "%WEIGHTS%" (
    echo [错误] 找不到预训练权重 %WEIGHTS%
    pause
    exit /b 1
)

REM ---------- 打印训练信息 ----------
echo ==============================================
echo   跌倒检测模型训练
echo ----------------------------------------------
echo   图片尺寸    : %IMG%
echo   批大小      : %BATCH%
echo   训练轮数    : %EPOCHS%
echo   数据配置    : %DATA%
echo   预训练权重  : %WEIGHTS%
echo   超参配置    : %HYP%
echo   早停耐心    : %PATIENCE% 轮
echo   设备        : %DEVICE%  ^(0=GPU, cpu=CPU^)
echo ==============================================
echo.

REM ---------- 开始训练 ----------
venv\Scripts\python.exe yolov5\train.py --img %IMG% --batch %BATCH% --epochs %EPOCHS% --data %DATA% --weights %WEIGHTS% --device %DEVICE% --workers %WORKERS% --hyp %HYP% --patience %PATIENCE% --cos-lr

echo.
echo ==============================================
echo   训练结束。
echo   权重位置: yolov5\runs\train\exp*\weights\
echo     - best.pt = 效果最好的权重
echo     - last.pt = 最后一轮的权重
echo ==============================================
pause
