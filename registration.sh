data='/data2/HBNcore/CMI_HBN_Data/MRI/RU/data/'
ants_out='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/'
ants_work='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/working/'
outpath='/data2/Projects/Jake/Registration_PEER'

cd $outpath

rm copy_raw.txt
rm copy_ants.txt

cd $data

##########################################################################
# Copy raw peer data

#for sub in $(ls);do
for sub in 'sub-5002891';do

	n=`find "$data/$sub/func" -iname '*peer*' -iname '*nii.gz*' |wc -l`

	if [ $n -gt "1" ]; then

	for nifti_file in $(ls $data'/'$sub'/'func/*.nii.gz);do

		if [[ $nifti_file == *"peer_run-1"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_1.nii.gz" >> $outpath/copy_raw.txt


		elif [[ $nifti_file == *"peer_run-2"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_2.nii.gz" >> $outpath/copy_raw.txt


		elif [[ $nifti_file == *"peer_run-3"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_3.nii.gz" >> $outpath/copy_raw.txt

		fi

	done

	fi

		mkdir $outpath'/'$sub

done

cat $outpath/copy_raw.txt | parallel -j 5

##########################################################################
# Copy C-PAC outputs for ANTs registration

cd $ants_out

#for sub in $(ls);do 
for sub in 'sub-5002891_ses-1';do

	deskull=$ants_out/$sub/anatomical_brain/*
	skull=$ants_out/$sub/anatomical_reorient/*

	echo "cp $deskull $outpath/no_skull.nii.gz" >> $outpath/copy_out.txt
	echo "cp $skull $outpath/skull.nii.gz" >> $outpath/copy_out.txt

done

cat $outpath/copy_out.txt | parallel -j 5

cd $ants_work

for sub in 'resting_preproc_sub-5002891_ses-1';do

	segments=$ants_work/$sub/seg_preproc_0/WM/WM_mask/segment_seg_2_maths.nii.gz

	echo "cp $segments $outpath/segments.nii.gz" >> $outpath/copy_working.txt

done

cat $outpath/copy_working.txt | parallel -j 5























