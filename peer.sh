# This script runs PEER on the participants of a given directory, assuming that they each have 3 PEER scans.

data='/data2/HBNcore/CMI_HBN_Data/MRI/RU/data_Backup'
outpath='/home/json/Desktop/PEER_bash'

rm command_list.txt
rm mri_template.txt
rm registration.txt

cd $data

for sub in $(ls);do

	if grep "$sub" "/home/json/Desktop/PEER_data/motion_test.txt"; then	

		echo $sub

		for nifti_file in $(ls $data'/'$sub'/'func/*.nii.gz);do

			if [[ $nifti_file == *"peer_run-1"* ]];then
				echo found peer 1
				echo -n "mcflirt -in $nifti_file -out $outpath/$sub/peer1_mcf.nii.gz" >> $outpath/command_list.txt
				echo ";fslroi $outpath/$sub/peer1_mcf.nii.gz $outpath/$sub/template_1.nii.gz 67 1" >> $outpath/command_list.txt

			elif [[ $nifti_file == *"peer_run-2"* ]];then
				echo -n "mcflirt -in $nifti_file -out $outpath/$sub/peer2_mcf.nii.gz" >> $outpath/command_list.txt
				echo ";fslroi $outpath/$sub/peer2_mcf.nii.gz $outpath/$sub/template_2.nii.gz 67 1" >> $outpath/command_list.txt
				echo found peer 2

			elif [[ $nifti_file == *"peer_run-3"* ]];then
				echo -n "mcflirt -in $nifti_file -out $outpath/$sub/peer3_mcf.nii.gz" >> $outpath/command_list.txt
				echo ";fslroi $outpath/$sub/peer3_mcf.nii.gz $outpath/$sub/template_3.nii.gz 67 1" >> $outpath/command_list.txt
				echo found peer 3

			fi

		done

		mkdir $outpath'/'$sub

	echo "echo $sub processing completed" >> $outpath/command_list.txt
	echo "mri_robust_template --mov $outpath/$sub/template_1.nii.gz $outpath/$sub/template_2.nii.gz $outpath/$sub/template_3.nii.gz --template $outpath/$sub/mean.nii.gz --satit" >> $outpath/mri_template.txt

	fi

done

cd $outpath

cat command_list.txt | parallel -j 25
cat mri_template.txt | parallel -j 25

for sub in $(ls);do

	echo "flirt -in $outpath/$sub/peer1_mcf.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER1_resampled -applyisoxfm 4" >> $outpath/registration.txt
	echo "flirt -in $outpath/$sub/peer2_mcf.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER2_resampled -applyisoxfm 4" >> $outpath/registration.txt
	echo "flirt -in $outpath/$sub/peer3_mcf.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER3_resampled -applyisoxfm 4" >> $outpath/registration.txt

done

cat registration.txt | parallel -j 25






