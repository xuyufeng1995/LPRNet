#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging

import torch

from model.lprnet import LPRNet, CHARS
from model.stnet import STNet
from utils.general import set_logging

logger = logging.getLogger(__name__)
set_logging()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/exp5/weights/best.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--img-size', default=(96, 48), help='the image size')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    opts = parser.parse_args()

    # 打印参数
    logger.info("args: %s" % opts)

    # Input
    img = torch.zeros((opts.batch_size, 3, opts.img_size[1], opts.img_size[0]))

    # 定义网络
    device = torch.device('cpu')
    model = LPRNet(class_num=len(CHARS), dropout_rate=opts.dropout_rate)
    logger.info("Build network is successful.")

    # Load weights
    ckpt = torch.load(opts.weights, map_location=device)

    # 加载网络
    model.load_state_dict(ckpt["model"])

    # 释放内存
    del ckpt

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opts.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'], output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('Export complete.')
