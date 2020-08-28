apt-get update
apt-get install -y --no-install-recommends ffmpeg libavcodec-dev libavformat-dev libswscale-dev build-essential pkg-config libglib2.0-dev libgtk2.0-dev python3-pip pkg-config cmake unzip git
unzip ./dockerbuild/opencv3.4.10-with-freetype.zip -d ./opencv
cd opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_opencv_flann=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_ml=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF .. && make -j$(nproc) && sudo make install # 编译安装opencv
cd ../../
unzip ./dockerbuild/darknet.zip -d ./darknet
cd C++ && mkdir build && cd build && cmake .. && make -j$(nproc)
cp libdark.so ../../  # 将动态编译库文件拷贝至与manage.py同级
cp prepare ../../     # 将prepare文件拷贝至与manage.py同级
pip3 install -i https://pypi.douban.com/simple/ -r requirements.txt --default-timeout=100
