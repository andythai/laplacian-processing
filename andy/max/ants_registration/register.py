import ants
import numpy as np
import nibabel as nib

print("START")
template_path = "processed/smartspim_25um_cropped.nii"
input_path0 = "240913_newscan/240913_new_scan_25downsampled_ch0.nii.gz"
input_path1 = "240913_newscan/240913_new_scan_25downsampled_ch1.nii.gz"
input_path2 = "240913_newscan/240913_new_scan_25downsampled_ch2.nii.gz"

output_path0 = "240913_newscan/output/ch0_registered.nii.gz"
output_path1 = "240913_newscan/output/ch1_registered.nii.gz"
output_path2 = "240913_newscan/output/ch2_registered.nii.gz"

print("READING")

moving_image0 = nib.load(input_path0).get_fdata()
moving_image1 = nib.load(input_path1).get_fdata()
moving_image2 = nib.load(input_path2).get_fdata()
fixed_image = nib.load(template_path).get_fdata()


print("PROCESSING")
moving_image0 = np.rot90(moving_image0, k=1, axes=(1, 2))  # Ch0
moving_image0 = np.flip(moving_image0, axis=0)
moving_image1 = np.rot90(moving_image1, k=1, axes=(1, 2))  # Ch1
moving_image1 = np.flip(moving_image1, axis=0)
moving_image2 = np.rot90(moving_image2, k=1, axes=(1, 2))  # Ch2
moving_image2 = np.flip(moving_image2, axis=0)
fixed_image = np.moveaxis(fixed_image, -1, 0)


print("CONVERT TO ANTS")
#fixed_image = ants.image_read(template_path)
#moving_image0 = ants.image_read(input_path0)
#moving_image1 = ants.image_read(input_path1)
#moving_image2 = ants.image_read(input_path2)
moving_image0 = ants.from_numpy(moving_image0)
moving_image1 = ants.from_numpy(moving_image1)
moving_image2 = ants.from_numpy(moving_image2)
fixed_image = ants.from_numpy(fixed_image)

print("INPUT")
print("Before registering")
result = ants.registration(fixed_image, moving_image2, type_of_transform = 'SyN' )
print("Registered ch2")
ch1 = ants.apply_transforms(fixed=fixed_image, moving=moving_image1,
                                      transformlist=result['fwdtransforms'])
print("Registered ch1")
ch0 = ants.apply_transforms(fixed=fixed_image, moving=moving_image0,
                                      transformlist=result['fwdtransforms'])
print("Registered ch0")
#result = result.numpy()
#np.save("ants_result.npy", result)
ants.image_write(result['warpedmovout'], output_path2)
print(type(ch1))
print(ch1.shape)
ants.image_write(ch1, output_path1)
ants.image_write(ch0, output_path0)

print("DONE")