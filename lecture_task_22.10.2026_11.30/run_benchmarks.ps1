$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & javac -Xlint:all BenchmarkSupport.java Task1.java Task2.java Task3.java Task4.java Task5.java Task6.java Task7.java BenchmarkTests.java
    if ($LASTEXITCODE -ne 0) { throw 'Java compilation failed' }
    $tests = @(& java -ea BenchmarkTests)
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }
    $tests | ForEach-Object { Write-Output $_ }
    [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot 'test_output.txt'),
        [string[]]$tests, (New-Object System.Text.UTF8Encoding($false)))

    # Run sequentially: competing benchmarks would distort the measurements.
    foreach ($task in 1..7) {
        $lines = @(& java -Xms512m -Xmx2g "Task$task")
        if ($LASTEXITCODE -ne 0) { throw "Task $task failed; output file not replaced" }
        $lines | ForEach-Object { Write-Output $_ }
        [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot "task${task}_output.txt"),
            [string[]]$lines, (New-Object System.Text.UTF8Encoding($false)))
    }
} finally {
    Pop-Location
}