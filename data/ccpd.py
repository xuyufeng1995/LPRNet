import os
from data.bases import ImageDataset, Img
from data.dataset import LPRDataSet


class CCPD(ImageDataset):
    def __init__(self, source_dir: str, cache_dir: str):
        super().__init__(source_dir, cache_dir)

        # from cache
        cached_imgs = self.load_cache()
        if not cached_imgs:
            cached_imgs = self.cache_labels(source_dir)

        if cached_imgs:
            if cached_imgs['source_dir'] == source_dir:
                self.im_files = cached_imgs['imgs']

    def cache_labels(self, source_dir):
        if os.path.isfile(source_dir):
            with open(source_dir) as f:
                for line in f.read().strip().splitlines():
                    img_path, label = line.split("\t")
                    filepath = os.path.join(os.path.dirname(source_dir), img_path)
                    # if (len(label) == 8 and label[2] in ["D", "F"]) or len(label) == 7:
                    self.im_files.append(Img(filepath, label))
        else:
            for filename in os.listdir(source_dir):
                filepath = os.path.join(source_dir, filename)
                img = self.load_img(filepath, filename)
                if img:
                    self.im_files.append(img)


        cache = {
            'source_dir': source_dir,
            'imgs': self.im_files
        }
        self.save_cache(cache)
        return cache

    def load_img(self, img_path: str, file_name: str):
        # 去后缀
        img_name, suffix = os.path.splitext(file_name)

        # 按 - 切分
        img_name_split = img_name.split('-')
        if len(img_name_split) >= 2:
            # 获得车牌部分
            number = img_name_split[1]
            # if (len(number) == 8 and number[2] in ["D", "F"]) or len(number) == 7:
            return Img(img_path, number)

        return None

def load_dataset(source_dir, cache_dir, img_size):
    train_dataset = LPRDataSet(CCPD(os.path.join(source_dir, "train.txt"), cache_dir), img_size)
    test_dataset = LPRDataSet(CCPD(os.path.join(source_dir, "test.txt"), cache_dir), img_size)
    return train_dataset, test_dataset
