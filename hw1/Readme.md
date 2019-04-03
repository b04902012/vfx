# How to execute

`$ python main.py [image_directory_name]`
This script will read the `.hdr_image_list.txt` file and the images in the directory, align the images, produce radiance map, and do tone-mapping.

You can choose to skip the radiance map generating process and just load the existent pickle file if you're not the first time running this script.

For the tone-mapping part, you can choose to run global operator, local operator or hybrid (both of them).