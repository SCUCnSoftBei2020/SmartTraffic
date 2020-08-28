#!/bin/bash
rm ./dockerbuild/darknet.zip
cd C++ && zip -r ../dockerbuild/darknet.zip * && cd ..
# shellcheck disable=SC2164
cd dockerbuild
sudo docker build -t godkillerxiao/cnsoft:0.1 .