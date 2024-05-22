import random
import shutil
import subprocess
import sys
from enum import Enum
from io import BytesIO
from typing import List, Tuple

import accelerate
import requests
import torch
import torch.distributed as dist
from detectron2.data.detection_utils import convert_PIL_to_numpy
from diffusers import DiffusionPipeline
from diffusers_interpret.generated_images import GeneratedImages
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as F
from transformers import AutoFeatureExtractor


def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
  """Logs git status to stdout."""
  subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
  subprocess.call('echo', shell=True, stdout=out_file)
  exclude_string = ''
  subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def truncate_caption(caption: str) -> str:
  """Truncate captions at periods and newlines."""
  caption = caption.strip('\n')
  trunc_index = caption.find('\n') + 1
  if trunc_index <= 0:
      trunc_index = caption.find('.') + 1
  if trunc_index > 0:
    caption = caption[:trunc_index]
  return caption


def pad_to_size(x, size=256):
  delta_w = size - x.size[0]
  delta_h = size - x.size[1]
  padding = (
    delta_w // 2,
    delta_h // 2,
    delta_w - (delta_w // 2),
    delta_h - (delta_h // 2),
  )
  new_im = ImageOps.expand(x, padding)
  return new_im


class RandCropResize(object):

  """
  Randomly crops, then randomly resizes, then randomly crops again, an image. Mirroring the augmentations from https://arxiv.org/abs/2102.12092
  """

  def __init__(self, target_size):
    self.target_size = target_size

  def __call__(self, img):
    img = pad_to_size(img, self.target_size)
    d_min = min(img.size)
    img = T.RandomCrop(size=d_min)(img)
    t_min = min(d_min, round(9 / 8 * self.target_size))
    t_max = min(d_min, round(12 / 8 * self.target_size))
    t = random.randint(t_min, t_max + 1)
    img = T.Resize(t)(img)
    if min(img.size) < 256:
      img = T.Resize(256)(img)
    return T.RandomCrop(size=self.target_size)(img)


class SquarePad(object):
  """Pads image to square.
  From https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
  """
  def __call__(self, image):
    max_wh = max(image.size)
    p_left, p_top = [(max_wh - s) // 2 for s in image.size]
    p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
    padding = (p_left, p_top, p_right, p_bottom)
    return F.pad(image, padding, 0, 'constant')


def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text.

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  """
  height = 12
  padding = 5
  effective_width = width - 2 * padding
  # Create a black image to draw text on.
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  draw = ImageDraw.Draw(cap_img)
  draw.text((0, 0), text, color, font=font or ImageFont.load_default())
  cap_img = F.convert_image_dtype(F.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img


def get_feature_extractor_for_model(model_name: str, image_size: int = 224, train: bool = True):
  print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
  feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
  return feature_extractor


def get_pixel_values_for_model(feature_extractor, img: Image.Image):
  pixel_values = feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
  return pixel_values


def save_checkpoint(state, is_best, filename='checkpoint'):
  torch.save(state, filename + '.pth.tar')
  if is_best:
    shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def accuracy(output, target, padding, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    mask = (target != padding).type(target.dtype)
    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / mask.sum()))
    return res


def get_params_count(model, max_name_len: int = 60):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
  padding = 70  # Hardcoded depending on desired amount of padding and separators.
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
    param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))
    
  def display_summary(self):
    entries = [" *"]
    entries += [meter.summary() for meter in self.meters]
    print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
    self.name = name
    self.fmt = fmt
    self.summary_type = summary_type
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def all_reduce(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    self.sum, self.count = total.tolist()
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)
  
  def summary(self):
    fmtstr = ''
    if self.summary_type is Summary.NONE:
      fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
      fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
      fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
      fmtstr = '{name} {count:.3f}'
    else:
      raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)

def mask_target_cls(image: torch.Tensor, target_cls_id: int, det_model, pipe: DiffusionPipeline) -> torch.Tensor:
      """
          use detect model to mask the target cls

          Args:
            image: image tensor
            target_cls_id: target class id
            det_model: detect model, default should be Eva-02
          Returns:
            return_outputs: List consisting of either str or List[PIL.Image.Image] objects, representing image-text interleaved model outputs.
      """
      if target_cls_id == -1 or det_model is None:
          
          return torch.ones_like(image, dtype=torch.bool)

      # clone and detach
      image = image.clone().detach()
      height, width, channel = image.shape
      # transform image to PIL type to adapt the input of detect model
      all_images = GeneratedImages(
          all_generated_images=[image],
          pipe=pipe,
          remove_batch_dimension=True,
          prepare_image_slider=False
      )

      image = torch.as_tensor(convert_PIL_to_numpy(
          all_images[-1], format="BGR").astype("float32").transpose(2, 0, 1))

      inputs = {"image": image, "height": height, "width": width}
      predictions = det_model([inputs])[0]
      # offload det_model to cpu for saving gpu memory
      accelerate.cpu_offload(det_model)

      instances = predictions['instances']
      pred_masks = instances.pred_masks
      pred_classes = instances.pred_classes
      
      mask = torch.any(pred_masks[pred_classes == target_cls_id], dim=0)
      
      return mask.unsqueeze(0).repeat(channel, 1, 1).permute(1, 2, 0)


def gradients_attribution(
    pred_logits: torch.Tensor,
    input_embeds: Tuple[torch.Tensor],
    det_model,
    pipe: DiffusionPipeline,
    retain_graph: bool = False,
    target_cls_id: int = -1,
) -> List[torch.Tensor]:
      # TODO: add description

      assert len(pred_logits.shape) == 3

      # get mask matrix for target class
      target_mask = mask_target_cls(pred_logits, target_cls_id, det_model, pipe)

      # Construct tuple of scalar tensors with all `pred_logits`
      # The code below is equivalent to `tuple_of_pred_logits = tuple(torch.flatten(pred_logits))`,
      #  but for some reason the gradient calculation is way faster if the tensor is flattened like this
      tuple_of_pred_logits = []
      for px, mx in zip(pred_logits, target_mask):
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
      # default use gradients of input_embeds
      for grad, inp in zip(grads, input_embeds):
        aggregated_grads.append(torch.norm(grad * inp, dim=-1))

      return aggregated_grads
