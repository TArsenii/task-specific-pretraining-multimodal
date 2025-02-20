from dataclasses import dataclass

from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU


@dataclass
class ConvBlockArgs:
    conv_one_in: int
    conv_one_out: int
    conv_one_kernel_size: int | tuple[int, int] = (3, 3)
    conv_one_stride: int | tuple[int, int] = (1, 1)
    conv_one_padding: int | tuple[int, int] = (1, 1)


class ConvBlock(Module):
    def __init__(
        self,
        conv_block_one_args: ConvBlockArgs,
        conv_block_two_args: ConvBlockArgs,
        batch_norm: bool = True,
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv_one = Conv2d(
            conv_block_one_args.conv_one_in,
            conv_block_one_args.conv_one_out,
            kernel_size=conv_block_one_args.conv_one_kernel_size,
            stride=conv_block_one_args.conv_one_stride,
            padding=conv_block_one_args.conv_one_padding,
        )
        self.conv_two = Conv2d(
            conv_block_two_args.conv_one_in,
            conv_block_two_args.conv_one_out,
            kernel_size=conv_block_two_args.conv_one_kernel_size,
            stride=conv_block_two_args.conv_one_stride,
            padding=conv_block_two_args.conv_one_padding,
        )
        self.relu = ReLU()
        self.do_batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_one = BatchNorm2d(conv_block_one_args.conv_one_out)
            self.batch_norm_two = BatchNorm2d(conv_block_two_args.conv_one_out)
        else:
            self.batch_norm_one, self.batch_norm_two = None, None

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.conv_one(tensor)
        if self.batch_norm_one is not None:
            tensor = self.batch_norm_one(tensor)
        tensor = self.relu(tensor)
        tensor = self.conv_two(tensor)
        if self.batch_norm_two is not None:
            tensor = self.batch_norm_two(tensor)
        tensor = self.relu(tensor)
        return tensor

    def running_stats(self, val: bool) -> None:
        self.batch_norm_one.track_running_stats = val
        self.batch_norm_two.track_running_stats = val
