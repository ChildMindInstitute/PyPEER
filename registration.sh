data='/data2/HBNcore/CMI_HBN_Data/MRI/RU/data/'
ants_out='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/'
ants_work='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/working/'
outpath='/data2/Projects/Jake/Registration_PEER'

cd $outpath

rm copy_raw.txt
rm copy_out.txt
rm copy_working.txt
rm anat_to_temp.txt
rm flirt.txt
rm file_convert.txt
rm transformations.txt

template='/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
template_skull='/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'

cd $data

##########################################################################
# Copy raw peer data

#for sub in $(ls);do
for sub in 'sub-5161675';do

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

#cat $outpath/copy_raw.txt | parallel -j 5

##########################################################################
# Copy C-PAC outputs for ANTs registration

cd $ants_out

#for sub in $(ls);do 
for sub in 'sub-5161675_ses-1';do

	deskull=$ants_out/$sub/anatomical_brain/*
	skull=$ants_out/$sub/anatomical_reorient/*
	fmean1=$ants_out/$sub/mean_functional/_scan_peer_run-1/*
	fmean2=$ants_out/$sub/mean_functional/_scan_peer_run-2/*
	fmean3=$ants_out/$sub/mean_functional/_scan_peer_run-3/*

	outsub=$(echo $sub | cut -f1 -d"_")

	echo "cp $deskull $outpath/$outsub/no_skull.nii.gz" >> $outpath/copy_out.txt
	echo "cp $skull $outpath/$outsub/skull.nii.gz" >> $outpath/copy_out.txt
	echo "cp $fmean1 $outpath/$outsub/fmean1.nii.gz" >> $outpath/copy_out.txt
	echo "cp $fmean2 $outpath/$outsub/fmean2.nii.gz" >> $outpath/copy_out.txt
	echo "cp $fmean3 $outpath/$outsub/fmean3.nii.gz" >> $outpath/copy_out.txt

done

cd $ants_work

for sub in 'resting_preproc_sub-5161675_ses-1';do

	outsub=$(echo $sub | cut -f3 -d"_")
	outsub=${outsub#*preproc_}

	segments=$ants_work/$sub/seg_preproc_0/WM/WM_mask/segment_seg_2_maths.nii.gz

	echo "cp $segments $outpath/$outsub/segments.nii.gz" >> $outpath/copy_working.txt

done

#cat $outpath/copy_out.txt | parallel -j 5
#cat $outpath/copy_working.txt | parallel -j 5

##########################################################################
# ANTs registration - anatomical to template
# Rate limiting step

cd $outpath

for subject in $(ls -d */);do

	echo "antsRegistration --collapse-output-transforms 0 --dimensionality 3 --initial-moving-transform [$template,$outpath/$subject/no_skull.nii.gz,0] --interpolation Linear --output [$outpath/$subject/transform, $outpath/$subject/transform_Warped.nii.gz] --transform Rigid[0.1] --metric MI[$template, $outpath/$subject/no_skull.nii.gz, 1, 32, Regular, 0.25] --convergence [1000x500x250x100, 1e-08, 10] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 1 --transform Affine[0.1] --metric MI[$template, $outpath/$subject/no_skull.nii.gz, 1, 32, Regular, 0.25] --convergence [1000x500x250x100, 1e-08, 10] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 1 --transform SyN[.1, 3.0, 0.0] --metric CC[$template_skull, $outpath/$subject/skull.nii.gz, 1, 4] --convergence [100x100x70x20, 1e-09, 15] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 6x4x2x1 --use-histogram-matching 1 --winsorize-image-intensities [.01, .99]" >> $outpath/anat_to_temp.txt

done

#cat $outpath/anat_to_temp.txt | parallel -j 20

##########################################################################
# Linear registration from functional to anatomical

for subject in $(ls -d */);do

	for file in $(ls $outpath/$subject);do

		if [[ $file = *"peer_1"* ]];then

			echo -n "flirt -in $outpath/$subject$file -ref $outpath/$subject/no_skull.nii.gz -out $outpath/$subject/func_to_anat1.nii.gz -omat $outpath/$subject/func_to_anat1.mat -cost corratio -dof 6 -interp trilinear;" >> $outpath/flirt.txt
			echo "flirt -in $outpath/$subject$file -ref $outpath/$subject/no_skull.nii.gz -out $outpath/$subject/func_to_anat_bbreg1.nii.gz -omat $outpath/$subject/func_to_anat_bbreg1.mat -cost bbr -wmseg $outpath/$subject/segments.nii.gz -dof 6 -init $outpath/$subject/func_to_anat1.mat -schedule '/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'" >> $outpath/flirt.txt

		elif [[ $file = *"peer_2"* ]];then

			echo -n "flirt -in $outpath/$subject$file -ref $outpath/$subject/no_skull.nii.gz -out $outpath/$subject/func_to_anat2.nii.gz -omat $outpath/$subject/func_to_anat2.mat -cost corratio -dof 6 -interp trilinear;" >> $outpath/flirt.txt
			echo "flirt -in $outpath/$subject$file -ref $outpath/$subject/no_skull.nii.gz -out $outpath/$subject/func_to_anat_bbreg2.nii.gz -omat $outpath/$subject/func_to_anat_bbreg2.mat -cost bbr -wmseg $outpath/$subject/segments.nii.gz -dof 6 -init $outpath/$subject/func_to_anat2.mat -schedule '/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'" >> $outpath/flirt.txt

		elif [[ $file = *"peer_3"* ]];then

			echo -n "flirt -in $outpath/$subject$file -ref $outpath/$subject/no_skull.nii.gz -out $outpath/$subject/func_to_anat3.nii.gz -omat $outpath/$subject/func_to_anat3.mat -cost corratio -dof 6 -interp trilinear;" >> $outpath/flirt.txt
			echo "flirt -in $outpath/$subject$file -ref $outpath/$subject/no_skull.nii.gz -out $outpath/$subject/func_to_anat_bbreg3.nii.gz -omat $outpath/$subject/func_to_anat_bbreg3.mat -cost bbr -wmseg $outpath/$subject/segments.nii.gz -dof 6 -init $outpath/$subject/func_to_anat3.mat -schedule '/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'" >> $outpath/flirt.txt

		fi

	done

done

#cat flirt.txt | parallel -j 20

##########################################################################
# Apply warps and use outputs for antsApplyTransforms

for subject in $(ls -d */);do

	for file in $(ls $outpath/$subject);do

		if [[ $file = *"anat1.nii.gz"* ]];then

			echo "" >> $outpath/file_convert.txt

		elif [[ $file = *"anat2.nii.gz"* ]];then

			echo "" >> $outpath/file_convert.txt

		elif [[ $file = *"anat3.nii.gz"* ]];then

			echo "" >> $outpath/file_convert.txt

		fi

	done

done

#cat file_convert.txt | parallel -j 10

##########################################################################
# Apply all transformations to get from functional to template space

for subject in $(ls -d */);do

	for file in $(ls $outpath/$subject);do

		if [[ $file = *"raw_peer_1"* ]];then

			echo "antsApplyTransforms --default-value 0 --dimensionality 3 --input-image-type 3 -v 1 --input $outpath/$subject/raw_peer_1.nii.gz --reference-image $template_skull --output $outpath/$subject/peer1_warped.nii.gz --transform $outpath/$subject/transform3Warp.nii.gz --transform $outpath/$subject/transform2Affine.mat --transform $outpath/$subject/transform1Rigid.mat --transform $outpath/$subject/transform0DerivedInitialMovingTranslation.mat --transform $outpath/$subject/itk_func_to_anat_bbreg1.txt" >> $outpath/transformations.txt

		elif [[ $file = *"raw_peer_2"* ]];then

			echo "antsApplyTransforms --default-value 0 --dimensionality 3 --input-image-type 3 -v 1 --input $outpath/$subject/raw_peer_2.nii.gz --reference-image $template_skull --output $outpath/$subject/peer2_warped.nii.gz --transform $outpath/$subject/transform3Warp.nii.gz --transform $outpath/$subject/transform2Affine.mat --transform $outpath/$subject/transform1Rigid.mat --transform $outpath/$subject/transform0DerivedInitialMovingTranslation.mat --transform $outpath/$subject/itk_func_to_anat_bbreg2.txt" >> $outpath/transformations.txt

		elif [[ $file = *"raw_peer_3"* ]];then

			echo "antsApplyTransforms --default-value 0 --dimensionality 3 --input-image-type 3 -v 1 --input $outpath/$subject/raw_peer_3.nii.gz --reference-image $template_skull --output $outpath/$subject/peer3_warped.nii.gz --transform $outpath/$subject/transform3Warp.nii.gz --transform $outpath/$subject/transform2Affine.mat --transform $outpath/$subject/transform1Rigid.mat --transform $outpath/$subject/transform0DerivedInitialMovingTranslation.mat --transform $outpath/$subject/itk_func_to_anat_bbreg3.txt" >> $outpath/transformations.txt

		fi

	done

done

cat $outpath/transformations.txt | parallel -j 10






# Manual validation step for sub-5002891

#antsRegistration --collapse-output-transforms 0 --dimensionality 3 -v 1 --initial-moving-transform ['/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz','/data2/Projects/Jake/Registration_PEER'/'sub-5002891'/'no_skull.nii.gz',0] --interpolation Linear --output ['/data2/Projects/Jake/Registration_PEER'/'sub-5002891'/transform, '/data2/Projects/Jake/Registration_PEER'/'sub-5002891'/transform_Warped.nii.gz] --transform Rigid[0.1] --metric MI['/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz', '/data2/Projects/Jake/Registration_PEER'/'sub-5002891'/'no_skull.nii.gz', 1, 32, Regular, 0.25] --convergence [1000x500x250x100, 1e-08, 10] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 1 --transform Affine[0.1] --metric MI['/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz', '/data2/Projects/Jake/Registration_PEER'/'sub-5002891'/'no_skull.nii.gz', 1, 32, Regular, 0.25] --convergence [1000x500x250x100, 1e-08, 10] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 1 --transform SyN[.1, 3.0, 0.0] --metric CC['/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz', '/data2/Projects/Jake/Registration_PEER'/'sub-5002891'/'skull.nii.gz', 1, 4] --convergence [100x100x70x20, 1e-09, 15] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 6x4x2x1 --use-histogram-matching 1 --winsorize-image-intensities [.01, .99]













