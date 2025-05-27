from segment_anything import sam_model_registry, SamPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

class BaseOUTLINE:
    def __init__(self, args):
        self.args = args
        self.outline_model_weight = self.args['outline_model_weight']
        self.sam = sam_model_registry['vit_l'](checkpoint=f'model/{self.outline_model_weight}')
        self.model = SamPredictor(self.sam)
    
    def get_outline(self, img, bbox):
        self.model.set_image(img)
        mask, _, _ = self.model.predict(
            box=bbox,
            multimask_output=False
        )
        return mask

class OUTLINE:
    def __init__(self, args):
        self.args = args
        self.outline_model_weight = self.args['outline_model_weight']
        self.model = SAM2ImagePredictor.from_pretrained(self.outline_model_weight)
    
    def get_outline(self, img, bbox):
        self.model.set_image(img)
        mask, _, _ = self.model.predict(
            box=bbox,
            multimask_output=False
        )
        return mask