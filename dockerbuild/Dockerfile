FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER MrZilinXiao <me@mrxiao.net>
ENV TZ "Asia/Shanghai"
ENV DOCKER_SRC=CnSoftBei2020
ENV DOCKER_HOME=/app
WORKDIR $DOCKER_HOME
COPY ./sources.list /etc/apt/
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libavcodec-dev libavformat-dev libswscale-dev build-essential pkg-config libglib2.0-dev libgtk2.0-dev python3-pip pkg-config cmake unzip git && rm -rf /var/cache/apk/* && rm -rf /var/lib/apt/lists/* && apt-get autoremove
COPY ./ ./
RUN unzip opencv3.4.10-with-freetype.zip -d ./opencv && rm opencv3.4.10-with-freetype.zip
RUN cd opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_opencv_flann=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_ml=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF .. && make -j$(nproc) && make install
RUN unzip darknet.zip -d ./darknet && rm darknet.zip
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN cd darknet && mkdir build && cd build && cmake .. && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH make -j$(nproc)
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN pip3 install -i https://pypi.douban.com/simple/ -r requirements-docker.txt --default-timeout=100 --no-cache-dir
EXPOSE 8000
RUN chmod u+x start_script
CMD ["/app/start_script"]