#/usr/bin/bash
for FILE in ./digit-recog-wavs/*; do
	FILE_NAME=$(basename $FILE)
	sox $FILE -r 16k -c 1 ./digit_recog_wavs_downsampled/$FILE_NAME
	#echo $WAV_FILE
done



