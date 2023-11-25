#!/usr/bin/env powershell
param(
    $command = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

# clean build of NativeOps if with build command
if ($command -eq "build") {
    python NativeOps/build.py clean
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    python NativeOps/build.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

# run tests
dotnet test
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# build App
Remove-Item -Recurse -Force publish -ErrorAction Ignore
dotnet publish App -r win-x64 -c Release -o publish -p:PublishSingleFile=true --self-contained true
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# modifi file stucture
Move-Item publish/runtimes/win-x64/native/*.dll publish/
Remove-Item -Recurse -Force publish/runtimes
Remove-Item -Force publish/appsettings.*.json
Remove-Item -Force publish/*.pdb
git rev-parse HEAD | Out-File -FilePath publish/commit.txt

# make zip
Compress-Archive -Path publish/* -DestinationPath publish/llm-sharp_win-x64.zip
