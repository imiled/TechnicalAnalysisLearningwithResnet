
vpath=$(pwd)

cd images/
mv state_N_* tmp/NN
mv state_BB* tmp/BB
mv state_NB* tmp/NB
mv state_SS* tmp/SS
mv state_SN* tmp/SN
mv state_ER* tmp/ER
rm -rf tmp/ER
rm -rf .ipynb_checkpoints

cd tmp/BB
shuf -n 40 -e * | xargs -i mv {} ../../imagestest/BB
cd ../NB
shuf -n 40 -e * | xargs -i mv {} ../../imagestest/NB
cd ../NN
shuf -n 40 -e * | xargs -i mv {} ../../imagestest/NN
cd ../SN
shuf -n 40 -e * | xargs -i mv {} ../../imagestest/SN
cd ../SS
shuf -n 40 -e * | xargs -i mv {} ../../imagestest/SS


cd $vpath
cd images/

ls tmp/BB | wc -l >> nbimagepblock.txt
ls tmp/NB | wc -l >> nbimagepblock.txt
ls tmp/NN | wc -l >> nbimagepblock.txt
ls tmp/SN | wc -l >> nbimagepblock.txt
ls tmp/SS | wc -l >> nbimagepblock.txt

var=$(datamash min 1 < nbimagepblock.txt)

cd tmp/BB/
shuf -n $var -e * | xargs -i mv {} ../../imagestrain/BB
cd ../NB/
shuf -n $var -e * | xargs -i mv {} ../../imagestrain/NB
cd ../NN/
shuf -n $var -e * | xargs -i mv {} ../../imagestrain/NN
cd ../SN/
shuf -n $var -e * | xargs -i mv {} ../../imagestrain/SN
cd ../SS/
shuf -n $var -e * | xargs -i mv {} ../../imagestrain/SS

cd $vpath
ls -a images/imagestrain
ls -a images/imagestest
