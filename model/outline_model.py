from segment_anything import sam_model_registry, SamPredictor

class OUTLINE:
    def __init__(self, args):
        self.args = args
        self.outline_model_weight = self.args['outline_model_weight']
        self.sam = sam_model_registry['vit_l'](checkpoint=f'model/{self.outline_model_weight}')
        self.model = SamPredictor(self.sam)
    
    def get_outline(self, img, bbox_lst):
        self.model.set_image(img)
        
        mask_lst = []
        for bbox in bbox_lst:
            mask, _, _ = self.model.predict(
                box=bbox,
                multimask_output=False
            )
            mask_lst.append(mask)
        return mask_lst