import os, sys, json, shutil
from pathlib import Path
import torch.utils.cpp_extension as torch_ext


if "include" in sys.argv:
    print(json.dumps(torch_ext.include_paths(cuda=True), indent=2))
    print(json.dumps(torch_ext.library_paths(cuda=True), indent=2))
    exit(0)

build_name = "llm_sharp_ops"
root = Path(__file__).parent
src = root / "src"


build_path = root / "build"
build_path.mkdir(parents=True, exist_ok=True)


sources = [str(x) for x in src.glob("*.cpp")] + [str(x) for x in src.glob("*.cu")]
include_paths = [str(src)]


if "clean" in sys.argv:
    shutil.rmtree(build_path)
    exit(0)


if torch_ext.IS_WINDOWS:
    import setuptools._distutils._msvccompiler as msvc
    vc_env = msvc._get_vc_env("x86_amd64")

    if not vc_env:
        raise Exception("Unable to find a compatible Visual Studio installation.")

    def append_env(env: str, paths: str):
        paths = paths.split(os.pathsep)
        for p in paths:
            if env not in os.environ:
                os.environ[env] = ""
            if len(p) and p not in os.environ[env]:
                os.environ[env] = p + os.pathsep + os.environ[env]

    append_env("path", vc_env.get("path", ""))
    append_env("include", vc_env.get("include", ""))
    append_env("lib", vc_env.get("lib", ""))


torch_ext.load(
    build_name,
    sources=sources,
    extra_include_paths=include_paths,
    build_directory=str(build_path),
    with_cuda=True,
    is_python_module=False,
)


runtimes_path = root / "runtimes"

if torch_ext.IS_WINDOWS:
    build_target = build_path / (build_name + torch_ext.LIB_EXT)
    runtimes_target = runtimes_path / "win-x64" / "native" / (build_name + torch_ext.CLIB_EXT)
    runtimes_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(build_target, runtimes_target)

elif torch_ext.IS_LINUX:
    build_target = build_path / (build_name + torch_ext.LIB_EXT)
    runtimes_target = runtimes_path / "linux-x64" / "native" / (build_name + torch_ext.CLIB_EXT)
    runtimes_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(build_target, runtimes_target)

else:
    raise Exception("Unsupported system")
