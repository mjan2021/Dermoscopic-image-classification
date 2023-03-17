from imports import *

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, default='../data/', help="Export Directory to store augmented images")
parser.add_argument("--data_dir", type=str, default='../data/', help="directory to read images from")
parser.add_argument("--meta", type=str, default='../csv/', help="number of epochs of training")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
opt = parser.parse_args()



def read_file(path):
    meta = pd.read_csv(path, index_col=False)
    return meta

df = read_file(opt.meta)
desc = df['dx'].value_counts()

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, 
                        zoom_range= 0.1, horizontal_flip= True, rescale=2)

aug_dir = opt.exp_dir
basedir = opt.data_dir+'/HAM10k/HAM10000_images/'
print(f"Folder Exists..") if os.path.exists(aug_dir) else os.mkdir(aug_dir)
save_dir = aug_dir
total_generated_images = 0
for key in desc.keys():
    print(key, desc[key])
    ratio = int(desc['nv']/ desc[key])
    print('Ratio to NV: ', ratio)
    print('Augmenataions Needed/Image', ratio)
    total_generated_images += ratio
    all_images = df[df['dx'] == key]['image_id'].values
    if(key == 'nv'):
        continue
    printonce= True
    # iterate over all images augment them, save them and insert them in our metadata frame
    for image_ in tqdm(all_images):
        if(len(df[df['dx'] == key]) > len(df[df['dx'] == 'nv'])):
            if printonce:
                print(key, 'datapoints = ', len(df[df['dx'] == key]), 'reached above nv skipping more augmentations..')
                printonce = False
            continue
    
        image_path =  basedir + image_ + '.jpg'
        image = load_img(image_path)
        image = np.expand_dims(img_to_array(image), axis= 0)
        generated = gen.flow(image)
        row = df[df['image_id'] == image_]
        dict_for_df = {
            'lesion_id':row.lesion_id.values[0], 'image_id':row.image_id.values[0], 
            'dx':row.dx.values[0], 'dx_type':row.dx_type.values[0] ,
            'age':row.age.values[0], 'sex':row.sex.values[0], 'localization':row.localization.values[0] 
        }
       
        for i in range(int(ratio)):
            aug_image= next(generated).astype(np.uint8)
            # save this image with an underscore
            # add this to metadata dataframe
            image_name= dict_for_df['image_id'] + '_' + str(i)
            fname = image_name.split('_')
            modify_name = fname[0]+'_'+fname[1]+'_'+fname[-1]
            dict_for_df['image_id'] = modify_name
            df = df.append(dict_for_df, ignore_index=True)
            plt.imsave(save_dir + modify_name + '.jpg', aug_image[0])
    
print(f"Total Images generated : {(total_generated_images)}")


# ===================================================================
# The Block  rename and copy the images to single directory, comment it out if not needed
print(f"Consolidating All Images...")

# rename the images
print(f"Renaming files to desired format...")
for file in tqdm(os.listdir(save_dir)):
    filename = file.split('_')
    renamed = filename[0]+'_'+filename[1]+'_'+filename[-1]
    os.rename(save_dir+file, save_dir+renamed)

#  Copy the images
for file in tqdm(os.listdir(basedir)):
    shutil.copy(basedir+file, save_dir)
    
print(f"Total Images: {len(os.listdir(save_dir))}")
# Modify and save the meta file
try:
    new_meta_file = f'{aug_dir}_Meta_all.csv'
    # df['new_image_id'] = df['image_id'].split()
    df.to_csv('../data/'+new_meta_file, index=False)
    print(f"New Meta File : {new_meta_file}")
except:
    print(f"FileError :: Error occured during meta file saving...")
#  ===================================================================