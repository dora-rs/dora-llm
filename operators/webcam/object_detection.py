#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import pyarrow as pa
import torch

from dora import DoraStatus

pa.array([])

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class Operator:
    """
    Infering object from images
    """

    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n")

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            return self.on_input(dora_event, send_output)
        return DoraStatus.CONTINUE

    def on_input(
        self,
        dora_input,
        send_output,
    ) -> DoraStatus:

        frame = dora_input["value"].to_numpy().reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
        frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
        results = self.model(frame)  # includes NMS
        arrays = pa.array(np.array(results.xyxy[0].cpu()).ravel())
        send_output("bbox", arrays, dora_input["metadata"])
        return DoraStatus.CONTINUE
