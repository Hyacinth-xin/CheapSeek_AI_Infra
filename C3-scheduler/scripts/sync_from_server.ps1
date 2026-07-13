$sshKey = "F:\scx\temp_key\.ssh\mig30"
$server = "mig30@39.107.68.147"
$port = "1130"
$baseDir = "e:\桌面\AI_Infra\dev\C3-scheduler"
$remoteDir = "~/CheapSeek_AI_Infra/C3-scheduler"

function Get-File($remotePath, $localPath) {
    $base64 = ssh -i $sshKey $server -p $port "base64 -w 0 $remotePath"
    $bytes = [Convert]::FromBase64String($base64)
    $content = [System.Text.Encoding]::UTF8.GetString($bytes)
    [System.IO.File]::WriteAllText($localPath, $content)
    Write-Host "$remotePath -> $localPath done"
}

Get-File "$remoteDir/kernel.py" "$baseDir\kernel.py"
Get-File "$remoteDir/hardware.py" "$baseDir\hardware.py"
Get-File "$remoteDir/graph.py" "$baseDir\graph.py"
Get-File "$remoteDir/strategy.py" "$baseDir\strategy.py"
Get-File "$remoteDir/memory_planner.py" "$baseDir\memory_planner.py"
Get-File "$remoteDir/export_dag.py" "$baseDir\export_dag.py"
Get-File "$remoteDir/graph_passes/__init__.py" "$baseDir\graph_passes\__init__.py"
Get-File "$remoteDir/graph_passes/fusion.py" "$baseDir\graph_passes\fusion.py"
Get-File "$remoteDir/tests/test_c33.py" "$baseDir\tests\test_c33.py"
Get-File "$remoteDir/tests/test_c34.py" "$baseDir\tests\test_c34.py"
Get-File "$remoteDir/scripts/create_test_model.py" "$baseDir\scripts\create_test_model.py"

Write-Host "All files synced from server!"
