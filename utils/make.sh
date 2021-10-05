clear

cd ./overlaps
rm *.so
make
cd ../

cd ./nms
rm *.so
make
cd ../


cd ./point_justify
rm *.so *.o
sh compile.sh
cd ../..

