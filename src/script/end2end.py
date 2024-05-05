import logging
from os import path
from typing import List, Optional, Tuple, Union

import accelerate
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers_interpret import StableDiffusionPipelineExplainer
from diffusers_interpret.data import (
    AttributionAlgorithm, AttributionMethods, BaseMimicPipelineCallOutput,
    PipelineExplainerForBoundingBoxOutput, PipelineExplainerOutput,
    PipelineImg2ImgExplainerForBoundingBoxOutputOutput,
    PipelineImg2ImgExplainerOutput)
from diffusers_interpret.generated_images import GeneratedImages
from modelscope.hub.snapshot_download import snapshot_download
from PIL.Image import Image
from detectron2.data.detection_utils import read_image, convert_PIL_to_numpy

# logger
logger = logging.getLogger(__name__)


class StableDiffusionPipelineDetExplainer(StableDiffusionPipelineExplainer):

    def __init__(self, pipe: DiffusionPipeline, verbose: bool = True, gradient_checkpointing: bool = False, det_model=None) -> None:
        super().__init__(pipe, verbose, gradient_checkpointing)
        self.det_model = det_model

    def _get_attributions(
        self,
        output: Union[PipelineExplainerOutput, PipelineExplainerForBoundingBoxOutput],
        attribution_method: AttributionMethods,
        tokens: List[List[str]],
        text_embeddings: torch.Tensor,
        init_image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None,
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int,
                                                          int], Tuple[int, int]]] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        **kwargs
    ) -> Union[
        PipelineExplainerOutput,
        PipelineExplainerForBoundingBoxOutput,
        PipelineImg2ImgExplainerOutput,
        PipelineImg2ImgExplainerForBoundingBoxOutputOutput
    ]:
        if self.verbose:
            print("Calculating token attributions... ", end='')

        target_cls_id = kwargs['target_cls_id']

        token_attributions = self.gradients_attribution(
            pred_logits=output.image,
            input_embeds=(text_embeddings,),
            attribution_algorithms=[
                attribution_method.tokens_attribution_method],
            explanation_2d_bounding_box=explanation_2d_bounding_box,
            target_cls_id=target_cls_id
        )[0].detach().cpu().numpy()

        output = self._post_process_token_attributions(
            output=output,
            tokens=tokens,
            token_attributions=token_attributions,
            consider_special_tokens=consider_special_tokens,
            clean_token_prefixes_and_suffixes=clean_token_prefixes_and_suffixes
        )

        if self.verbose:
            print("Done!")

        return output

    def _mask_target_cls(self, image: torch.Tensor, target_cls_id: int) -> torch.Tensor:
        """
            use detect model to mask the target cls
        """
        if target_cls_id == -1 or self.det_model is None:
            # 返回一个identity矩阵
            return torch.ones_like(image, dtype=torch.bool)

        # clone and detach
        image = image.clone().detach()
        height, width = image.shape[:2]
        # transform image to PIL type to adapt the input of detect model
        all_images = GeneratedImages(
            all_generated_images=[image],
            pipe=self.pipe,
            remove_batch_dimension=True,
            prepare_image_slider=False
        )

        image = torch.as_tensor(convert_PIL_to_numpy(
            all_images[-1], format="BGR").astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.det_model([inputs])[0]
        # offload det_model to cpu for saving gpu memory
        accelerate.cpu_offload(self.det_model)

        instances = predictions['instances']
        pred_masks = instances.pred_masks
        pred_classes = instances.pred_classes
        return torch.any(pred_masks[pred_classes == target_cls_id])

    def gradients_attribution(
        self,
        pred_logits: torch.Tensor,
        input_embeds: Tuple[torch.Tensor],
        attribution_algorithms: List[AttributionAlgorithm],
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int,
                                                          int], Tuple[int, int]]] = None,
        retain_graph: bool = False,
        target_cls_id: int = -1
    ) -> List[torch.Tensor]:
        # TODO: add description

        assert len(pred_logits.shape) == 3
        if explanation_2d_bounding_box:
            upper_left, bottom_right = explanation_2d_bounding_box
            pred_logits = pred_logits[upper_left[0]
                : bottom_right[0], upper_left[1]: bottom_right[1], :]

        assert len(input_embeds) == len(attribution_algorithms)

        # get mask matrix for target class
        traget_mask = self._mask_target_cls(pred_logits, target_cls_id)

        # Construct tuple of scalar tensors with all `pred_logits`
        # The code below is equivalent to `tuple_of_pred_logits = tuple(torch.flatten(pred_logits))`,
        #  but for some reason the gradient calculation is way faster if the tensor is flattened like this
        tuple_of_pred_logits = []
        for px, mx in zip(pred_logits, traget_mask):
            for py, my in zip(px, mx):
                for pz, mz in zip(py, my):
                    if mz:
                        tuple_of_pred_logits.append(pz)
        tuple_of_pred_logits = tuple(tuple_of_pred_logits)

        # get the sum of back-prop gradients for all predictions with respect to the inputs
        if torch.is_autocast_enabled():
            # FP16 may cause NaN gradients https://github.com/pytorch/pytorch/issues/40497
            # TODO: this is still an issue, the code below does not solve it
            with torch.autocast(input_embeds[0].device.type, enabled=False):
                grads = torch.autograd.grad(
                    tuple_of_pred_logits, input_embeds, retain_graph=retain_graph)
        else:
            grads = torch.autograd.grad(
                tuple_of_pred_logits, input_embeds, retain_graph=retain_graph)

        if torch.isnan(grads[-1]).any():
            raise RuntimeError(
                "Found NaNs while calculating gradients. "
                "This is a known issue of FP16 (https://github.com/pytorch/pytorch/issues/40497).\n"
                "Try to rerun the code or deactivate FP16 to not face this issue again."
            )

        # Aggregate
        aggregated_grads = []
        for grad, inp, attr_alg in zip(grads, input_embeds, attribution_algorithms):

            if attr_alg == AttributionAlgorithm.GRAD_X_INPUT:
                aggregated_grads.append(torch.norm(grad * inp, dim=-1))
            elif attr_alg == AttributionAlgorithm.MAX_GRAD:
                aggregated_grads.append(grad.abs().max(-1).values)
            elif attr_alg == AttributionAlgorithm.MEAN_GRAD:
                aggregated_grads.append(grad.abs().mean(-1).values)
            elif attr_alg == AttributionAlgorithm.MIN_GRAD:
                aggregated_grads.append(grad.abs().min(-1).values)
            else:
                raise NotImplementedError(
                    f"aggregation type `{attr_alg}` not implemented")

        return aggregated_grads


class Trainer:
    def __init__(self, workspace: str = '/mnt/workspace', eva_id: str = 'zacbi2023/eva02', sd_id: str = 'AI-ModelScope/stable-diffusion-2-1',):
        self.base_path = workspace
        self.model_path = path.join(workspace, 'model')
        self.data_path = path.join(workspace, "data")
        self.eva_id = eva_id
        self.sd_id = sd_id

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.download_model()
        self.prepare_stable_diffusion()
        self.prepare_eva()

    def download_model(self):
        model_ids = [self.sd_id, self.eva_id]
        for model_id in model_ids:
            if not path.exists(path.join(self.model_path, model_id)):
                snapshot_download(model_id, cache_dir=self.model_path)

    def prepare_stable_diffusion(self):
        self.sd_pipeline = AutoPipelineForText2Image.from_pretrained(
            path.join(self.model_path, self.sd_id), torch_dtype=self.torch_dtype).to('cuda')

    def prepare_eva(self):
        # replace with your eva02 config path
        eva_base_path = path.join(self.model_path, self.eva_id)
        eva_coco_config_rpath = 'projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py'
        eva_config_path = path.join(eva_base_path, eva_coco_config_rpath)

        # replace with your eva02 weights path
        eva_coco_weights_rpth = 'checkpoints/eva02_L_coco_seg_sys_o365.pth'
        eva_weights_path = path.join(eva_base_path, eva_coco_weights_rpth)

        custum_cfg = ['MODEL.RETINANET.SCORE_THRESH_TEST', 0.5,
                      'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,
                      'MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH', 0.5,
                      'DATASETS.TEST', [],
                      'MODEL.WEIGHTS', eva_weights_path]
        eva_cfg = LazyConfig.load(eva_config_path)
        LazyConfig.apply_overrides(
            eva_cfg, [f"{key}={value}" for key, value in zip(custum_cfg[::2], custum_cfg[1::2])])

        self.eva = instantiate(eva_cfg.model).to(self.device)
        DetectionCheckpointer(self.eva).load(eva_weights_path)
        self.eva.eval()

    def infer(self, prompt: str = 'A cat sits on the chair', target_cls_id=15):
        """
            target_cls_id = 15 为 cat
            # 1. sd 向前传播
            # 2. eva图像分割获取目标区域mask
            # 3. sd-interpret根据mask进行归因
        """

        explainer = StableDiffusionPipelineDetExplainer(
            self.sd_pipeline, det_model=self.det_model)
        with torch.autocast('cuda'):
            output = explainer(
                prompt,
                num_inference_steps=50,
                n_last_diffusion_steps_to_consider_for_attributions=1,
                target_cls_id=target_cls_id
            )
        return output


def main():
    trainer = Trainer()
    output = trainer.infer()
    return output


if __name__ == "__main__":
    main()
