# 后台启动 auto-evolve，输出重定向到日志文件
$logFile = "D:\prompt-ops\use-cases\parent-child-coach\results\auto_evolve_kimi.log"
$python = "python"
$script = "D:\prompt-ops\use-cases\parent-child-coach\auto_evolve\run_auto_evolve.py"
$workdir = "D:\prompt-ops\use-cases\parent-child-coach"

# 清空旧日志
Set-Content -Path $logFile -Value "" -Encoding utf8

# 启动后台进程
$proc = Start-Process -FilePath $python -ArgumentList $script -WorkingDirectory $workdir -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru -NoNewWindow

Write-Output "Started PID=$($proc.Id)"
Write-Output "Log: $logFile"
Write-Output "To check progress: Get-Content $logFile -Tail 30"
