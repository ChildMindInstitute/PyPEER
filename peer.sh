# This script runs PEER on the participants of a given directory, assuming that they each have 3 PEER scans.

data='/data2/HBNcore/CMI_HBN_Data/MRI/RU/data_Backup'
outpath='/home/json/Desktop/PEER_bash'

cd $data

for sub in $(ls);do

	cd $data

	if grep "$sub" "/home/json/Desktop/PEER_data/participants_RU_peer_reduced.txt"; then	

		echo $sub

		for nifti_file in $(ls $data'/'$sub'/'func/*.nii.gz);do

			if [[ $nifti_file == *"peer_run-1"* ]];then
				peer1=$nifti_file
#				echo "mcflirt -in $peer1 -out $outpath/$sub'/peer1_mcf'" >> $outpath/command_list.txt
#				echo "flirt -in $peer1 -ref $peer1 -out $outpath/$sub/PEER1_resampled -applyisoxfm 4" >> $outpath/command_list.txt
				echo "cp $peer1 $outpath/$sub" >> $outpath/command_list.txt
				echo "fslroi $peer1 $outpath/$sub/template_1.nii.gz 67 1" >> $outpath/command_list.txt

			elif [[ $nifti_file == *"peer_run-2"* ]];then
				peer2=$nifti_file
#				echo "mcflirt -in $peer2 -out $outpath/$sub'/peer2_mcf'" >> $outpath/command_list.txt
#				echo "flirt -in $peer2 -ref $peer1 -out $outpath/$sub/PEER2_resampled -applyisoxfm 4" >> $outpath/command_list.txt
				echo "cp $peer2 $outpath/$sub" >> $outpath/command_list.txt
				echo "fslroi $peer2 $outpath/$sub/template_2.nii.gz 67 1" >> $outpath/command_list.txt

			elif [[ $nifti_file == *"peer_run-3"* ]];then
				peer3=$nifti_file
#				echo "mcflirt -in $peer3 -out $outpath/$sub'/peer3_mcf'" >> $outpath/command_list.txt
#				echo "flirt -in $peer3 -ref $peer1 -out $outpath/$sub/PEER3_resampled -applyisoxfm 4" >> $outpath/command_list.txt
				echo "cp $peer3 $outpath/$sub" >> $outpath/command_list.txt
				echo "fslroi $peer3 $outpath/$sub/template_3.nii.gz 67 1" >> $outpath/command_list.txt

			fi

		done

		mkdir $outpath'/'$sub

	echo "cd $outpath/$sub; var=$(ls -1 | wc -l); if ((echo $var < 6));then rm -r $outpath/$sub; fi" >> $outpath/remove_incompletes.txt
	echo "echo $sub processing completed" >> $outpath/command_list.txt
	echo "mri_robust_template --mov $outpath/$sub/template_1.nii.gz $outpath/$sub/template_2.nii.gz $outpath/$sub/template_3.nii.gz --template $outpath/$sub/mean.nii.gz --satit" >> $outpath/mri_template.txt

	fi

	cd $data

done

cd $outpath

cat command_list.txt | parallel -j 25
cat mri_template.txt | parallel -j 25

# CURRENTLY NEED TO MANUALLY RENAME PARTICIPANTS THAT HAVE POOR DATA (e.g. more than one of PEER1, incomplete PEER2, no PEER3)

for sub in $(ls);do

	echo "flirt -in $outpath/$sub/${sub}_task-peer_run-1_bold.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER1_resampled -applyisoxfm 4" >> $outpath/registration.txt
	echo "flirt -in $outpath/$sub/${sub}_task-peer_run-2_bold.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER2_resampled -applyisoxfm 4" >> $outpath/registration.txt
	echo "flirt -in $outpath/$sub/${sub}_task-peer_run-3_bold.nii.gz -ref $outpath/$sub/'mean.nii.gz' -out $outpath/$sub/PEER3_resampled -applyisoxfm 4" >> $outpath/registration.txt

done

cat registration.txt | parallel -j 25






