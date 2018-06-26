ants_out='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC'
outpath='/data2/Projects/Jake/Human_Brain_Mapping'

rm $outpath/resample.txt
rm $outpath/eye_extract.txt

for sub in $(ls $ants_out);do

	sub=$(echo $sub | cut -f1 -d"_")

#	mkdir $outpath/$sub

#	echo "flirt -in $ants_out/${sub}_ses-1/motion_correct_to_standard/_scan_peer_run-1/${sub}_task-peer_run-1_bold_calc_tshift_resample_volreg_antswarp.nii.gz -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_3mm.nii.gz -out $outpath/$sub/peer1_resampled.nii.gz -applyisoxfm 3" >> $outpath/resample.txt

#	echo "flirt -in $ants_out/${sub}_ses-1/motion_correct_to_standard/_scan_peer_run-2/${sub}_task-peer_run-2_bold_calc_tshift_resample_volreg_antswarp.nii.gz -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_3mm.nii.gz -out $outpath/$sub/peer2_resampled.nii.gz -applyisoxfm 3" >> $outpath/resample.txt

#	echo "flirt -in $ants_out/${sub}_ses-1/motion_correct_to_standard/_scan_peer_run-3/${sub}_task-peer_run-3_bold_calc_tshift_resample_volreg_antswarp.nii.gz -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_3mm.nii.gz -out $outpath/$sub/peer3_resampled.nii.gz -applyisoxfm 3" >> $outpath/resample.txt

#	echo "cp $ants_out/${sub}_ses-1/motion_correct_to_standard/_scan_peer_run-1/${sub}_task-peer_run-1_bold_calc_tshift_resample_volreg_antswarp.nii.gz $outpath/$sub/peer1.nii.gz" >> $outpath/copy.txt

#	echo "cp $ants_out/${sub}_ses-1/motion_correct_to_standard/_scan_peer_run-2/${sub}_task-peer_run-2_bold_calc_tshift_resample_volreg_antswarp.nii.gz $outpath/$sub/peer2.nii.gz" >> $outpath/copy.txt

#	echo "cp $ants_out/${sub}_ses-1/motion_correct_to_standard/_scan_peer_run-3/${sub}_task-peer_run-3_bold_calc_tshift_resample_volreg_antswarp.nii.gz $outpath/$sub/peer3.nii.gz" >> $outpath/copy.txt

	echo "fslroi $outpath/$sub/peer1.nii.gz $outpath/$sub/peer1_eyes_sub.nii.gz 13 65 72 36 1 34 0 -1" >> $outpath/eye_extract.txt

#	echo "fslroi $outpath/$sub/peer2.nii.gz $outpath/$sub/peer2_eyes_sub.nii.gz 13 65 72 36 1 34 0 -1" >> $outpath/eye_extract.txt

#	echo "fslroi $outpath/$sub/peer3.nii.gz $outpath/$sub/peer3_eyes_sub.nii.gz 13 65 72 36 1 34 0 -1" >> $outpath/eye_extract.txt

done

cat $outpath/eye_extract.txt | parallel -j 15
