cd C++
rm -r build
mkdir build
cd build
cmake ..
make retest -j12
make prepare -j12
cd ..
cp retest ../
cp prepare ../