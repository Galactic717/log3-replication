# Черга реплікації LOG3 Vault як завдання планувальника Windows (старт системи, перезапуск після збою).
# Парольна фраза не потрібна: run передає вже зашифровані об'єкти. Токен вузла — файлом (LOG3_NODE_TOKEN_FILE).
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1 `
#       -Exe C:\log3\venv\Scripts\log3-vault.exe -Repo C:\ProgramData\log3-vault -TokenFile C:\ProgramData\log3\node.token
#   Видалення: Unregister-ScheduledTask -TaskName LOG3-Vault -Confirm:$false
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$Repo,
    [string]$TokenFile = "",
    [int]$Interval = 30,
    [string]$TaskName = "LOG3-Vault"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Exe)) { throw "Не знайдено $Exe" }
$envPart = "set LOG3_VAULT_REPO=$Repo"
if ($TokenFile) { $envPart += "&& set LOG3_NODE_TOKEN_FILE=$TokenFile" }
$cmd = "/c $envPart&& `"$Exe`" run --daemon --interval $Interval --json >> `"$Repo\service.log`" 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmd
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "Завдання $TaskName зареєстровано й запущено; журнал: $Repo\service.log"
