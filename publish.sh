#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# clean build of NativeOps if with build command
if [ $# -eq 1 ] && [ $1 = "build" ]; then
    python NativeOps/build.py clean
    python NativeOps/build.py
fi

# run tests
dotnet test

# build App
rm -rf publish
dotnet publish App -r linux-x64 -c Release -o publish -p:PublishSingleFile=true --self-contained true

# modifi file stucture
mv publish/runtimes/linux-x64/native/*.so publish/
rm -rf publish/runtimes
rm -rf publish/appsettings.*.json
rm -rf publish/*.pdb
git rev-parse HEAD > publish/commit.txt

# make zip
zip -j -r publish/llm-sharp_linux-x64.zip publish/*
