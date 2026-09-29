@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
setlocal
title DiffGet 图片差异提取 - http://127.0.0.1:9426
cd /d %~dp0

rem ---- 检测 Python ----
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"
if not defined PY (
  echo [错误] 未检测到 Python，请先安装 Python 3.9+ 并勾选 "Add Python to PATH"
  echo        下载地址: https://www.python.org/downloads/
  pause
  exit /b 1
)

rem ---- 检测并安装依赖 ----
%PY% -c "import flask, PIL, numpy" >nul 2>nul
if errorlevel 1 (
  echo [提示] 首次运行，正在安装依赖 flask / pillow / numpy ...
  %PY% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if errorlevel 1 %PY% -m pip install -r requirements.txt
  %PY% -c "import flask, PIL, numpy" >nul 2>nul
  if errorlevel 1 (
    echo [错误] 依赖安装失败，请手动执行: %PY% -m pip install -r requirements.txt
    pause
    exit /b 1
  )
)

echo.
echo ============================================================
echo    DiffGet · 图片差异提取工具
echo    服务地址: http://127.0.0.1:9426
echo    浏览器将自动打开 ^(若未打开请手动访问上述地址^)
echo    关闭本窗口即停止服务
echo ============================================================
echo.
%PY% web_app.py
echo.
echo [服务已停止]
pause
