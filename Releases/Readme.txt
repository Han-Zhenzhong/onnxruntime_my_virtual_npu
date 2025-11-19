在根目录下执行如下命令收集 pre_release_cpp 内容保存到 PreRelease_<时间搓> 下
- chmod +x ./pre_package_cpp_sdk.sh
- ./pre_package_cpp_sdk.sh
将 pre_release_cpp 内容放到 Releases/ 下
- cp -rf PreRelease_<时间搓>/cpp Releases/<pre_release>/

打包 python 内容
- cp build/Release/dist/*.whl Releases/<pre_release>/python

将 Releases/<pre_release>/ 打包为 tgz 放到 Releases/<release>/下
