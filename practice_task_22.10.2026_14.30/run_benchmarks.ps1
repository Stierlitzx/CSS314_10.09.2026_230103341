param(
    [ValidateSet('all', '1', '2', '3', '4', '5')]
    [string]$Task = 'all'
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & javac -Xlint:all PracticeSupport.java Task1.java Task2.java Task3.java Task4.java Task5.java PracticeTests.java
    if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
    $tests = @(& java -ea -Xmx1g PracticeTests)
    if ($LASTEXITCODE -ne 0) { throw 'Correctness tests failed' }
    $tests | ForEach-Object { Write-Output $_ }
    [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot 'test_output.txt'),
        [string[]]$tests, (New-Object System.Text.UTF8Encoding($false)))
    $tasks = if ($Task -eq 'all') { 1..5 } else { @([int]$Task) }
    # Keep benchmarks sequential; concurrent CPU experiments invalidate their timings.
    foreach ($number in $tasks) {
        $lines = @(& java -Xms1g -Xmx3g "Task$number")
        if ($LASTEXITCODE -ne 0) { throw "Task $number failed; raw output not replaced" }
        $lines | ForEach-Object { Write-Output $_ }
        [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot "task${number}_output.txt"),
            [string[]]$lines, (New-Object System.Text.UTF8Encoding($false)))
    }
} finally {
    Pop-Location
}