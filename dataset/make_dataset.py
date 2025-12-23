import sys
import shutil
import os
from datasets.VisA import Visa_dataset
from datasets.MVTec import Mvtec_dataset
from datasets.BTAD import BTAD_dataset
from datasets.MPDD import MPDD_dataset
from datasets.MVTec3D import Mvtec3D_dataset
from datasets.RESC import RESC_dataset
from datasets.BrasTS import BrasTS_dataset
from datasets.VOC import Ade_dataset


def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def process_dataset(dataset_cls, src_root, des_root, id_start=0, binary=True, to_255=True):

    move(des_root)
    dataset = dataset_cls(src_root)
    return dataset.make_VAND(binary=binary, to_255=to_255, des_path_root=des_root, id=id_start)


if __name__ == "__main__":
    id_counter = 0


    datasets_config = [
        {
            "name": "visa",   # https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
            "class": Visa_dataset,
            "src": "/SOLUTION/Anomaly_detection_project/visa",
            "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/visa"
        }
        # {
        #     "name": "mvtec",  # https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
        #     "class": Mvtec_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/MVTecAD",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas"
        # }
        # {
        #     "name": "BTAD",  # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
        #     "class": BTAD_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/BeanTech_AD",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/BTAD"
        # },
        # {
        #     "name": "MPDD", # https://github.com/stepanje/MPDD
        #     "class": MPDD_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/MPDD",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/MPDD"
        # },
        # {
        #     "name": "mvtec3D",   # https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads
        #     "class": Mvtec3D_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/mvtec3D",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/mvtec3D"
        # },
        # {
        #     "name": "RESC",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
        #     "class": RESC_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/RESC",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/RESC"
        # },
        # {
        #     "name": "BrasTS",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
        #     "class": BrasTS_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/BrasTS",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/BrasTS"
        # },
        # {
        #     "name": "Ade20K",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
        #     "class": Ade_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/Ade",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/Ade"
        # },
        # {
        #     "name": "VOC",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
        #     "class": Ade_dataset,
        #     "src": "/SOLUTION/Anomaly_detection_project/dataset/VOC",
        #     "des": "/SOLUTION/Defect_detection_pcb/dataset/dictas/VOC"
        # },

    ]

    for config in datasets_config:
        print(f"Processing {config['name']}...")
        id_counter = process_dataset(
            dataset_cls=config["class"],
            src_root=config["src"],
            des_root=config["des"],
            id_start= 0
        )
        print(f"Finished {config['name']}, next ID: {id_counter}")

    