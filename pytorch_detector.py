"""
Module to run MegaDetector v5, a PyTorch YOLOv5 (Ultralytics) animal detection model,
on images.
"""

#%% Imports

import torch
import numpy as np
import traceback

from run_detector import CONF_DIGITS, COORD_DIGITS, FAILURE_INFER
import ct_utils

from ultralytics.data.augment import LetterBox, ToTensor
from ultralytics.models import YOLO
from ultralytics.utils.ops import scale_coords
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results, Boxes, Probs

print(f'Using PyTorch version {torch.__version__}')


#%% Classes

class PTDetector:

    IMAGE_SIZE = 1280  
    STRIDE = 64

    def __init__(self, model_path: str, force_cpu: bool = False, use_model_native_classes= False):
        self.device = 'cpu'
        if not force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            try:
                if torch.backends.mps.is_built and torch.backends.mps.is_available():
                    self.device = 'mps'
            except AttributeError:
                pass
        try:
            self.model = PTDetector._load_model(model_path, self.device)
        except Exception as e:
            # In a very esoteric scenario where an old version of YOLOv5 is used to run
            # newer models, we run into an issue because the "Model" class became
            # "DetectionModel".  New YOLOv5 code handles this case by just setting them
            # to be the same, so doing that via monkey-patch doesn't seem *that* rude.
            if "Can't get attribute 'DetectionModel'" in str(e):
                print('Forward-compatibility issue detected, patching')
                from models import yolo
                yolo.DetectionModel = yolo.Model                
                self.model = PTDetector._load_model(model_path, self.device)                
            else:
                raise
        if (self.device != 'cpu'):
            print('Sending model to GPU')
            self.model.to(self.device)
            
        self.printed_image_size_warning = False        
        self.use_model_native_classes = use_model_native_classes
        self.draw_boxes = True
        self.letterbox = LetterBox(new_shape=(PTDetector.IMAGE_SIZE, PTDetector.IMAGE_SIZE), stride=PTDetector.STRIDE, auto=True)
        self.to_tensor = ToTensor()
        

    @staticmethod
    def _load_model(model_pt_path, device):
        return YOLO(model_pt_path, verbose=True).to(device)

    def _preprocess(self, image):
        aug_image = self.letterbox(image=image)
        aug_image = self.to_tensor(aug_image)
        return aug_image.unsqueeze(dim=0).contiguous().to(device=self.device)

    def _draw_bounding_box(self, results, image, detection_threshold):#, class_names):
        annotator = Annotator(image)
        for det in results["detections"]:
            if det['conf'] > detection_threshold:
                annotator.box_label(self._convert_xywh_to_xyxy(det['bbox']), f"{det['category']} {det['conf']*100.0:.2f}%")
        results["annotated_image"] = annotator.result()
        return results

    def _convert_xywh_to_xyxy(self, xywh: list):
        xc = xywh[0]
        yc = xywh[1]
        w  = xywh[2]
        h  = xywh[3]

        x1 = xc - w / 2
        x2 = xc + w / 2
        y1 = yc - h / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]

    def _convert_xyxy_to_xywh(self, xyxy: list):
        x1 = xyxy[0]
        y1 = xyxy[1]
        x2 = xyxy[2]
        y2 = xyxy[3]

        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2
        yc = y1 + h / 2
        return [xc, yc, w, h]

    def _postprocess(self, detections: Results, image, aug_img_shape, detection_threshold, **kwargs):
        boxes: Boxes | None = detections.boxes
        class_names = detections.names
        results = None
        if boxes:
            scaled_boxes = scale_coords(
                aug_img_shape, boxes.xyxy.clone().reshape([-1,2,2]), image.shape[:2]
            ).round().reshape([-1,4])
            labels = boxes.cls
            confidences = boxes.conf

            dets = []
            for box, label, confidence in zip(scaled_boxes, labels, confidences):
                dets.append({'category': class_names[int(label)], 'conf': round(confidence.item(), 2), 'bbox': self._convert_xyxy_to_xywh(box.tolist())})
            results = {
                "number of detections": len(dets),
                #"boxes": scaled_boxes,
                #"labels": labels,
                #"confidences": confidences * 100.00,
                "detections": dets,
                "annotated_image": None,
            }

            if self.draw_boxes:
                results = self._draw_bounding_box(
                    results=results,
                    image=image,
                    detection_threshold=detection_threshold
                    #class_names=class_names
                )
        return results

    def generate_detections_one_image(self, img_original, image_id, detection_threshold, image_size=None):
        """Apply the detector to an image.

        Args:
            img_original: the PIL Image object with EXIF rotation taken into account
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """

        result = {
            'file': image_id
        }
        detections = []
        max_conf = 0.0
        img_original = np.asarray(img_original)
        aug_image = self._preprocess(image=img_original)
        with torch.no_grad():
            detections = self.model(aug_image, conf=detection_threshold)[0]
        return self._postprocess(detections=detections, image=img_original, detection_threshold=detection_threshold, aug_img_shape=aug_image.shape[2:])

if __name__ == '__main__':
    # for testing

    import visualization_utils as viz_utils

    model_file = "<path to the model .pt file>"
    im_file = "test_images/test_images/island_conservation_camera_traps_palau_cam10a_cam10a12122018_palau_cam10a12122018_20181108_174532_rcnx1035.jpg"

    detector = PTDetector(model_file)
    image = viz_utils.load_image(im_file)

    res = detector.generate_detections_one_image(image, im_file, detection_threshold=0.00001)
