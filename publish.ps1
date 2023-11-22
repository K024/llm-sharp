#!/usr/bin/env powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

# clean build of NativeOps
python NativeOps/build.py clean
python NativeOps/build.py

# run tests
dotnet test

# build App
Remove-Item -Recurse -Force publish -ErrorAction Ignore
dotnet publish App -r win-x64 -c Release -o publish -p:PublishSingleFile=true --self-contained true

# modifi file stucture
Move-Item publish/runtimes/win-x64/native/*.dll publish/
Remove-Item -Recurse -Force publish/runtimes
Remove-Item -Force publish/appsettings.*.json
git rev-parse HEAD | Out-File -FilePath publish/commit.txt

# make zip
Compress-Archive -Path publish/* -DestinationPath publish/llm-sharp_win-x64.zip
