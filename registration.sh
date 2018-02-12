data='/data2/HBNcore/CMI_HBN_Data/MRI/RU/data/'
ants='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/'
outpath='/data2/Projects/Jake/Registration_PEER'

cd $data

rm $outpath/command.txt

##########################################################################
# Copy raw peer data

#for sub in $(ls);do
for sub in 'sub-5002891';do

	n=`find "$data/$sub/func" -iname '*peer*' -iname '*nii.gz*' |wc -l`

	if [ $n -gt "1" ]; then

	num=0

	for nifti_file in $(ls $data'/'$sub'/'func/*.nii.gz);do

		if [[ $nifti_file == *"peer_run-1"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_1.nii.gz" >> $outpath/command.txt


		elif [[ $nifti_file == *"peer_run-2"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_2.nii.gz" >> $outpath/command.txt


		elif [[ $nifti_file == *"peer_run-3"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_3.nii.gz" >> $outpath/command.txt

		fi

	done

	fi

		mkdir $outpath'/'$sub

done

cat $outpath/command.txt | parallel -j 5

##########################################################################
# Copy C-PAC outputs for ANTs registration































