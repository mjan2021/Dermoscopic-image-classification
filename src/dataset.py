
from imports import *

class SkinCancer(Dataset):
    """Skin Cancer Dataset."""

    def __init__(self, root_dir, meta, transform=None):
        """
        Args:
            root_dir (string): Path to root directory containing images
            meta_file (string): Path to csv file containing images metadata (image_id, class)

            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        
        self.root_dir = root_dir
        self.meta = meta
        self.transform = transform

        self.df = pd.read_csv(self.meta)
        self.image_paths = self.df['image_path'].to_list()  
        self.image_ids = self.df['image_id'].to_list()
        self.classes = sorted(self.df['dx'].unique().tolist())
        self.classes_all = self.df['dx'].tolist()

        
        self.class_id = {i:j for i, j in enumerate(self.classes)}
        self.class_to_id = {value:key for key,value in self.class_id.items()}
        
        self.class_count =  self.df['dx'].value_counts().to_dict()
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, -1]
        label = self.df.iloc[idx, 2]
        image = Image.open(img_path)
        image_tensor = self.transform(image)
        label_id = torch.tensor(self.class_to_id[str(label)])
        return image_tensor, label_id
