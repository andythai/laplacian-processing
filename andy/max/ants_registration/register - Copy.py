import ants
print("START")
template_path = "processed/smartspim_25um_cropped_brightened.nii"
input_path0 = "rescanned/processed/rescanned_eric_25um_C0.nii"
input_path1 = "rescanned/processed/rescanned_eric_25um_C1.nii"
input_path2 = "rescanned/processed/rescanned_eric_25um_C2.nii"

output_path0 = "rescanned/output/rescanned_result_brightened_C0.nii.gz"
output_path1 = "rescanned/output/rescanned_result_brightened_C1.nii.gz"
output_path2 = "rescanned/output/rescanned_result_brightened_C2.nii.gz"

print("READING")
fixed_image = ants.image_read(template_path)
print("TEMPLATE")
moving_image0 = ants.image_read(input_path0)
moving_image1 = ants.image_read(input_path1)
moving_image2 = ants.image_read(input_path2)

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