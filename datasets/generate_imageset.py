import os
import sys
import glob
from PIL import Image
from tqdm import tqdm

DATASETS = ['IC15', 'IC13', 'MSRA_TD500',
            'HRSC2016', 'DOTA', 'UCAS_AOD', 'NWPU_VHR' ,
            'GaoFenShip', 'GaoFenAirplane', 
            'VOC']


def bmpToJpg(file_path):
   for fileName in tqdm(os.listdir(file_path)):
       newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
       im = Image.open(os.path.join(file_path,fileName))
       rgb = im.convert('RGB')      
       rgb.save(os.path.join(file_path,newFileName))

def del_bmp(root_dir=None):
    file_list = os.listdir(root_dir)
    for f in file_list:
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            if f.endswith(".BMP") or f.endswith(".bmp"):
                os.remove(file_path)
                print( "File removed! " + file_path)
        elif os.path.isdir(file_path):
            del_bmp(file_path)




def generate_imagets(dataset, ROOT_dir):
    assert dataset in DATASETS, 'Not supported dataset'
    if dataset  == 'DOTA':
        # For DOTA
        train_img_path = os.path.join(ROOT_dir, "DOTA/trainsplit/images" )
        val_img_path =  os.path.join(ROOT_dir, "DOTA/valsplit/images" )
        set_file =  os.path.join(ROOT_dir, 'DOTA/trainval.txt')
        files= sorted(glob.glob(os.path.join(train_img_path, '**.*' ))) + sorted(glob.glob(os.path.join(val_img_path, '**.*' )))
        with open(set_file,'w') as f:
            for file in files:
                img_path, filename = os.path.split(file)
                name, extension = os.path.splitext(filename)
                if extension in ['.jpg', '.bmp','.png']:
                    f.write(os.path.join(file)+'\n')

    elif dataset in ['IC13', 'IC15', 'MSRA_TD500', 'GaoFenShip']:
        # For IC13
        # train_img_dir =  os.path.join(ROOT_dir, "ICDAR13/train/images" )
        # val_img_dir =  os.path.join(ROOT_dir, "ICDAR13/val/images" )
        # trainset =  os.path.join(ROOT_dir, 'ICDAR13/train.txt')
        # valset =  os.path.join(ROOT_dir, 'ICDAR13/test.txt')

        # For IC15
        # train_img_dir =  os.path.join(ROOT_dir, "ICDAR15/train" )
        # val_img_dir =  os.path.join(ROOT_dir, "ICDAR15/val" )
        # trainset =  os.path.join(ROOT_dir, 'ICDAR15/train.txt')
        # valset =  os.path.join(ROOT_dir, 'ICDAR15/test.txt')

        # For MSRA_TD500
        train_img_dir =  os.path.join(ROOT_dir, "MSRA_TD500/train" )
        val_img_dir =  os.path.join(ROOT_dir, "MSRA_TD500/test" )
        trainset =  os.path.join(ROOT_dir, 'MSRA_TD500/train.txt')
        valset =  os.path.join(ROOT_dir, 'MSRA_TD500/test.txt')         

        # For GaoFenShip
        # train_img_dir =  os.path.join(ROOT_dir, "data/ship/train" )
        # val_img_dir =  os.path.join(ROOT_dir, "data/ship/val" )
        # trainset =  os.path.join(ROOT_dir, 'data/ship/train.txt')
        # valset =  os.path.join(ROOT_dir, 'data/ship/test.txt')

        for set_file, im_dir in zip([trainset, valset], [train_img_dir, val_img_dir]):
            with open(set_file,'w') as f:
                if dataset in ['IC13', 'IC15']:
                    files = glob.glob(os.path.join(im_dir, '**.jpg*' ))
                elif dataset == 'GaoFenShip':
                    files = glob.glob(os.path.join(im_dir, '**.tiff*' ))
                elif dataset == 'MSRA_TD500':
                    files = glob.glob(os.path.join(im_dir, '**.JPG*' ))
                else:
                    raise NotImplementedError
                for file in files:
                    f.write(file+'\n')
    

    
    elif  dataset in ['HRSC2016', 'UCAS_AOD', 'VOC', 'NWPU_VHR']:
        # root_dir = 'NWPU_VHR' 
        # trainset = os.path.join(ROOT_dir,'NWPU_VHR/ImageSets/train.txt')
        # valset   =  os.path.join(ROOT_dir, 'NWPU_VHR/ImageSets/test.txt')
        # testset   =  os.path.join(ROOT_dir, 'NWPU_VHR/ImageSets/test.txt')
        # img_dir =  os.path.join(ROOT_dir, 'NWPU_VHR/AllImages')
        # label_dir =  os.path.join(ROOT_dir, 'NWPU_VHR/Annotations')

        root_dir = 'HRSC2016' 
        trainset = os.path.join(ROOT_dir,'HRSC2016/ImageSets/train.txt')
        valset   =  os.path.join(ROOT_dir, 'HRSC2016/ImageSets/test.txt')
        testset   =  os.path.join(ROOT_dir, 'HRSC2016/ImageSets/test.txt')
        img_dir =  os.path.join(ROOT_dir, 'HRSC2016/FullDataSet/AllImages')
        label_dir =  os.path.join(ROOT_dir, 'HRSC2016/FullDataSet/Annotations')

        # root_dir = 'UCAS_AOD' 
        # trainset = os.path.join(ROOT_dir,'UCAS_AOD/ImageSets/train.txt')
        # valset   =  os.path.join(ROOT_dir, 'UCAS_AOD/ImageSets/test.txt')
        # testset   =  os.path.join(ROOT_dir, 'UCAS_AOD/ImageSets/test.txt')
        # img_dir =  os.path.join(ROOT_dir, 'UCAS_AOD/AllImages')
        # label_dir =  os.path.join(ROOT_dir, 'UCAS_AOD/Annotations')


        for dataset in [trainset, valset, testset]:
            with open(dataset,'r') as f:
                names = f.readlines()
                if DATASET in ['HRSC2016', 'NWPU_VHR']:
                    paths = [os.path.join(img_dir,x.strip()+'.jpg\n') for x in names]
                elif DATASET == 'UCAS_AOD':
                    paths = [os.path.join(img_dir,x.strip()+'.png\n') for x in names]
                with open(os.path.join(ROOT_dir + '/' + root_dir,os.path.split(dataset)[1]), 'w') as fw:
                    fw.write(''.join(paths))


if __name__ == '__main__':
    DATASET = 'MSRA_TD500'
    ROOT_dir = '/data-input/Rotated-Cascade-RetinaNet'
    generate_imagets(DATASET, ROOT_dir)



