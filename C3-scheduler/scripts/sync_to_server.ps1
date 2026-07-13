$sshKey = "F:\scx\temp_key\.ssh\mig30"
$server = "mig30@39.107.68.147"
$port = "1130"
$baseDir = "e:\桌面\AI_Infra\dev\C3-scheduler"
$targetDir = "~/CheapSeek_AI_Infra/C3-scheduler"

function Send-File($localPath, $remotePath) {
    $content = Get-Content $localPath -Raw
    $base64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($content))
    $cmd = "echo '$base64' | base64 -d > $remotePath && echo '$remotePath done'"
    ssh -i $sshKey $server -p $port $cmd
}

ssh -i $sshKey $server -p $port "mkdir -p $targetDir/tests $targetDir/scripts $targetDir/graph_passes"

Send-File "$baseDir\kernel.py" "$targetDir/kernel.py"
Send-File "$baseDir\hardware.py" "$targetDir/hardware.py"
Send-File "$baseDir\graph.py" "$targetDir/graph.py"
Send-File "$baseDir\strategy.py" "$targetDir/strategy.py"
Send-File "$baseDir\memory_planner.py" "$targetDir/memory_planner.py"
Send-File "$baseDir\export_dag.py" "$targetDir/export_dag.py"
Send-File "$baseDir\graph_passes\__init__.py" "$targetDir/graph_passes/__init__.py"
Send-File "$baseDir\graph_passes\fusion.py" "$targetDir/graph_passes/fusion.py"
Send-File "$baseDir\tests\test_c33.py" "$targetDir/tests/test_c33.py"
Send-File "$baseDir\tests\test_c34.py" "$targetDir/tests/test_c34.py"
Send-File "$baseDir\scripts\create_test_model.py" "$targetDir/scripts/create_test_model.py"

echo "All files sent!"
