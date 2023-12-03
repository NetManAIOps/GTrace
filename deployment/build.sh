echo "---------> Downloading dependencies..."
cd anomaly_detection/src/tracegnn/models/gtrace/cache
git clone https://github.com/microsoft/vcpkg.git --depth=1

echo "---------> Building dependencies..."
rm -r build
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

echo "---------> Finished!"
