data='/data2/HBNcore/CMI_HBN_Data/MRI/CBIC/data'
#outpath='/home/json/Desktop/PEER_bash'
outpath='/data2/Projects/Jake/CBIC'

rm $outpath/command_list.txt
rm $outpath/mri_template.txt
rm $outpath/registration.txt

cd $data

for sub in $(ls);do

#	if grep "$sub" "/home/json/Desktop/PEER_data/testing_on_new.txt"; then	

	n=`find "$data/$sub/func" -iname '*peer*' -iname '*nii.gz*' |wc -l`

	if [ $n -gt "1" ]; then

		num=0

		for nifti_file in $(ls $data'/'$sub'/'func/*.nii.gz);do

			if [[ $nifti_file == *"peer_run-1"* ]];then
				echo -n "mcflirt -in $nifti_file -out $outpath/$sub/peer1_mcf.nii.gz" >> $outpath/command_list.txt
				echo ";fslroi $outpath/$sub/peer1_mcf.nii.gz $outpath/$sub/template_1.nii.gz 67 1" >> $outpath/command_list.txt
				

			elif [[ $nifti_file == *"peer_run-2"* ]];then
				echo -n "mcflirt -in $nifti_file -out $outpath/$sub/peer2_mcf.nii.gz" >> $outpath/command_list.txt
				echo ";fslroi $outpath/$sub/peer2_mcf.nii.gz $outpath/$sub/template_2.nii.gz 67 1" >> $outpath/command_list.txt

			elif [[ $nifti_file == *"peer_run-3"* ]];then
				echo -n "mcflirt -in $nifti_file -out $outpath/$sub/peer3_mcf.nii.gz" >> $outpath/command_list.txt
				echo ";fslroi $outpath/$sub/peer3_mcf.nii.gz $outpath/$sub/template_3.nii.gz 67 1" >> $outpath/command_list.txt

			fi

		done

		mkdir $outpath'/'$sub

	if [ "$n" -eq "2" ]; then
		
		echo -n "mri_robust_template --mov $outpath/$sub/template_1.nii.gz $outpath/$sub/template_2.nii.gz --template $outpath/$sub/mean.nii.gz --satit" >> $outpath/mri_template.txt

		echo ";echo $sub registration template complete" >> $outpath/mri_template.txt

	elif [ "$n" -eq "3" ]; then

		echo -n "mri_robust_template --mov $outpath/$sub/template_1.nii.gz $outpath/$sub/template_2.nii.gz $outpath/$sub/template_3.nii.gz --template $outpath/$sub/mean.nii.gz --satit" >> $outpath/mri_template.txt

		echo ";echo $sub registration template complete" >> $outpath/mri_template.txt

	fi

	fi

#	fi

done

cd $outpath

cat command_list.txt | parallel -j 20
cat mri_template.txt | parallel -j 20

for sub in $(ls);do

#	if grep "$sub" "/home/json/Desktop/PEER_data/testing_on_new.txt"; then

		echo "flirt -in $outpath/$sub/peer1_mcf.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER1_resampled -applyisoxfm 4" >> $outpath/registration.txt
		echo "flirt -in $outpath/$sub/peer2_mcf.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER2_resampled -applyisoxfm 4" >> $outpath/registration.txt
		echo "flirt -in $outpath/$sub/peer3_mcf.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER3_resampled -applyisoxfm 4" >> $outpath/registration.txt

#	fi

done

cat registration.txt | parallel -j 20

cd '/home/json/Desktop/peer'

echo "Beginning python script for PEER"

python3 peer.py


