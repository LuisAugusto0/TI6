# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    batch_size=1,  # batch size
    workers=4,  # number of dataloader workers
):
    # ----------------------------------------------------------
    # 🔧 CONFIGURAÇÕES INICIAIS (original)
    # ----------------------------------------------------------
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 🧠 CARREGAMENTO DO MODELO (original)
    # ----------------------------------------------------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # ----------------------------------------------------------
    # 🧱 DATALOADER (original)
    # ----------------------------------------------------------
    bs = batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ----------------------------------------------------------
    # 🧮 FUNÇÃO DE CSV (original)
    # ----------------------------------------------------------
    csv_path = save_dir / "predictions.csv"

    def write_to_csv(image_name, prediction, confidence):
        data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    # ----------------------------------------------------------
    # ⚡ AQUECIMENTO (original)
    # ----------------------------------------------------------
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # ----------------------------------------------------------
    # [MOD] Setup para paralelismo (producer-consumer + ThreadPoolExecutor)
    # ----------------------------------------------------------
    import threading  # [MOD]
    import queue as Queue  # [MOD]
    from concurrent.futures import ThreadPoolExecutor  # [MOD]


    # executor que fará a conversão numpy->tensor (pré-processamento) em background
    max_workers = min(workers, os.cpu_count())
    LOGGER.info(f"Using {max_workers} workers for background pre-processing")  # [MOD]
    executor = None  # [MOD] default 4 ou cpu_count
    if workers > 0:
        executor = ThreadPoolExecutor(max_workers=max_workers)  # [MOD]
        # buffer/queue para comunicação entre produtor e consumidor
        preproc_queue = Queue.Queue(maxsize=64)  # [MOD] limite para evitar encher memória

    # função de pré-processamento (unificada) -> usada tanto pelo producer quanto no modo webcam/vídeo
    def preprocess_image(im):  # [MOD]
        im_t = torch.from_numpy(im).to(model.device)
        im_t = im_t.half() if model.fp16 else im_t.float()
        im_t /= 255.0
        if len(im_t.shape) == 3:
            im_t = im_t[None]
        return im_t

    # produtor: lê o dataset e envia (future, im0, path) para a fila
    # coloca um sentinel (None) no fim para indicar término
    def producer_task():
        try:
            for path, im, im0s, vid_cap, s in dataset:
                # só produz pré-processamento para modo image + batch_size>1
                if not webcam and getattr(dataset, "mode", "") == "image" and batch_size > 1:
                    fut = executor.submit(preprocess_image, im)
                    # bloco se fila cheia (backpressure)
                    preproc_queue.put((fut, im0s.copy(), path))
                else:
                    # para modos diferentes, colocamos item "raw" para processamento direto (None future)
                    preproc_queue.put((None, im0s.copy(), path, im, vid_cap, s))
        finally:
            # sinal de fim
            preproc_queue.put(None)

   

        
    # iniciar o produtor (só se vamos usar batching de imagens) [MOD]
    producer_thread = None
    use_image_batching = (not webcam) and getattr(dataset, "mode", "") == "image" and batch_size > 1  # [MOD]
    use_workers = use_image_batching and (workers > 0)
    LOGGER.info(f"Image batching is {'ON' if use_image_batching else 'OFF'}, Workers: {workers} ({'using' if use_workers else 'not using'} background pre-processing)")  # [MOD]
    if use_image_batching and use_workers:
        producer_thread = threading.Thread(target=producer_task, daemon=True)
        producer_thread.start()
    
        

    # ----------------------------------------------------------
    # ----------------- LOOP PRINCIPAL -------------------------
    # ----------------------------------------------------------
    # Consumidor: se estamos com batching de imagens, consumimos da fila por batch.
    # Caso contrário, usamos o fluxo original por item.
    try:
        if use_image_batching and use_workers:  # [MOD] modo batch com producer-consumer
            # enquanto houver dados do producer
            finished = False
            while not finished:
                # formar um batch
                batch_futures = []
                batch_im0s = []
                batch_paths = []

                # coleta batch_size itens (ou menos, se sentinel aparecer)
                for _ in range(batch_size):
                    item = preproc_queue.get()
                    if item is None:
                        finished = True
                        break
                    fut, im0s_item, path_item = item
                    batch_futures.append(fut)
                    batch_im0s.append(im0s_item)
                    batch_paths.append(path_item)

                if not batch_futures:
                    break  # nada a processar (final)

                # Espera os resultados das futures (pré-process do CPU) -> consegue sobrepor com producer
                im_tensors = [f.result() for f in batch_futures]  # aguardando pré-processamento
                im_batch = torch.cat(im_tensors, dim=0)

                # Inferência e NMS com dt (original)
                with dt[1]:
                    visualize_path = increment_path(save_dir / Path(batch_paths[0]).stem, mkdir=True) if visualize else False
                    pred = model(im_batch, augment=augment, visualize=visualize_path)

                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # métricas de tempo por batch/imagem (mantive seu log original)
                batch_n = len(pred) if isinstance(pred, (list, tuple)) else 1
                inf_batch_ms = dt[1].dt * 1e3
                nms_batch_ms = dt[2].dt * 1e3
                inf_per_img_ms = inf_batch_ms / max(batch_n, 1)
                nms_per_img_ms = nms_batch_ms / max(batch_n, 1)

                # pós-processamento por item (reaproveita todo o seu código original)
                for i, det in enumerate(pred):
                    seen += 1
                    p = Path(batch_paths[i])
                    im0 = batch_im0s[i]
                    s = f"{p.name}: "
                    save_path = str(save_dir / p.name)
                    txt_path = str(save_dir / "labels" / p.stem)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    imc = im0.copy() if save_crop else im0
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    if len(det):
                        det[:, :4] = scale_boxes(im_batch.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                            confidence_str = f"{float(conf):.2f}"

                            if save_csv:
                                write_to_csv(p.name, names[c], confidence_str)

                            if save_txt:
                                coords = (
                                    (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    if save_format == 0
                                    else (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                                )
                                line = (cls, *coords, conf) if save_conf else (cls, *coords)
                                with open(f"{txt_path}.txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

                            if save_img or save_crop or view_img:
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(
                                    xyxy,
                                    imc,
                                    file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                                    BGR=True,
                                )

                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == "Linux" and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)
                    if save_img:
                        cv2.imwrite(save_path, im0)
                    # Log por imagem: tempos médios do batch (inferência e NMS) por imagem
                    LOGGER.info(
                        f"{s}{'' if len(det) else '(no detections), '}"
                        f"inf {inf_per_img_ms:.1f}ms/img, nms {nms_per_img_ms:.1f}ms/img"
                    )

                # resumo do batch
                LOGGER.info(
                    f"Batch {batch_n} imgs: "
                    f"inference {inf_batch_ms:.1f}ms total ({inf_per_img_ms:.1f}ms/img), "
                    f"NMS {nms_batch_ms:.1f}ms total ({nms_per_img_ms:.1f}ms/img)"
                )

            # fim while: produtor sinalizou None e fila esvaziou
            # aguarda término do produtor
            if producer_thread is not None:
                producer_thread.join(timeout=1.0)
            # encerra executor
            executor.shutdown(wait=True)
        elif use_image_batching and not use_workers:  # [MOD] modo batch sem producer-consumer
            LOGGER.info("⚠️  WARNING: NÚMERO DE WORKERS = 0, USANDO MODO BATCHING SEM PRÉ-PROCESSAMENTO ASSÍNCRONO")

            images_buf, im0s_buf, paths_buf = [], [], []

            for path, im, im0s, vid_cap, s in dataset:

                # PRÉ-PROCESSAMENTO (com dt[0])
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()
                    im /= 255.0
                    if len(im.shape) == 3:
                        im = im[None]
                    # Se for modo imagem → acumula para batch
                    if not webcam and dataset.mode == "image" and batch_size > 1:
                        images_buf.append(im)
                        im0s_buf.append(im0s.copy())
                        paths_buf.append(path)
                    
                        # Quando atingir o batch size → processa todas de uma vez
                        if len(images_buf) == batch_size:
                            im_batch = torch.cat(images_buf, dim=0)
                        
                            with dt[1]:
                                        visualize_path = increment_path(save_dir / Path(paths_buf[0]).stem, mkdir=True) if visualize else False
                                        pred = model(im_batch, augment=augment, visualize=visualize_path)

                            with dt[2]:
                                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                        
                            # Medidas de tempo do batch atual (inferência e NMS)
                            batch_n = len(pred) if isinstance(pred, (list, tuple)) else 1
                            inf_batch_ms = dt[1].dt * 1e3
                            nms_batch_ms = dt[2].dt * 1e3
                            inf_per_img_ms = inf_batch_ms / max(batch_n, 1)
                            nms_per_img_ms = nms_batch_ms / max(batch_n, 1)

                            for i, det in enumerate(pred):
                                            seen += 1
                                            p = Path(paths_buf[i])
                                            im0 = im0s_buf[i]
                                            s = f"{p.name}: "
                                            save_path = str(save_dir / p.name)
                                            txt_path = str(save_dir / "labels" / p.stem)
                                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                                            imc = im0.copy() if save_crop else im0
                                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                                            if len(det):
                                                det[:, :4] = scale_boxes(im_batch.shape[2:], det[:, :4], im0.shape).round()
                                                for c in det[:, 5].unique():
                                                    n = (det[:, 5] == c).sum()
                                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                                                for *xyxy, conf, cls in reversed(det):
                                                    c = int(cls)
                                                    label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                                                    confidence_str = f"{float(conf):.2f}"

                                                    if save_csv:
                                                        write_to_csv(p.name, names[c], confidence_str)

                                                    if save_txt:
                                                        coords = (
                                                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                                            if save_format == 0
                                                            else (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                                                        )
                                                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                                                        with open(f"{txt_path}.txt", "a") as f:
                                                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                                                    if save_img or save_crop or view_img:
                                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                                    if save_crop:
                                                        save_one_box(
                                                            xyxy,
                                                            imc,
                                                            file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                                                            BGR=True,
                                                        )

                                            im0 = annotator.result()
                                            if view_img:
                                                if platform.system() == "Linux" and p not in windows:
                                                    windows.append(p)
                                                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                                                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                                                cv2.imshow(str(p), im0)
                                                cv2.waitKey(1)
                                            if save_img:
                                                cv2.imwrite(save_path, im0)
                                            # Log por imagem: tempos médios do batch (inferência e NMS) por imagem
                                            LOGGER.info(
                                                f"{s}{'' if len(det) else '(no detections), '}"
                                                f"inf {inf_per_img_ms:.1f}ms/img, nms {nms_per_img_ms:.1f}ms/img"
                                            )

                            # Resumo do batch: tempos totais e médias por imagem
                            LOGGER.info(
                                f"Batch {batch_n} imgs: "
                                f"inference {inf_batch_ms:.1f}ms total ({inf_per_img_ms:.1f}ms/img), "
                                f"NMS {nms_batch_ms:.1f}ms total ({nms_per_img_ms:.1f}ms/img)"
                            )

                            images_buf.clear()
                            im0s_buf.clear()
                            paths_buf.clear()
                            continue  # pula resto do loop, pois já processou o batch
            # ----------------------------------------------------------
            # [ADDED] FLUSH FINAL DO BUFFER (processa último lote incompleto)
            # Motivo: garantir que imagens restantes (< batch_size) também sejam inferidas
            # ----------------------------------------------------------
            if (not webcam) and getattr(dataset, "mode", "") == "image" and batch_size > 1 and len(images_buf) > 0:
                im_batch = torch.cat(images_buf, dim=0)

                with dt[1]:
                    visualize_path = increment_path(save_dir / Path(paths_buf[0]).stem, mkdir=True) if visualize else False
                    pred = model(im_batch, augment=augment, visualize=visualize_path)

                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Medidas de tempo do batch atual (inferência e NMS)
                batch_n = len(pred) if isinstance(pred, (list, tuple)) else 1
                inf_batch_ms = dt[1].dt * 1e3
                nms_batch_ms = dt[2].dt * 1e3
                inf_per_img_ms = inf_batch_ms / max(batch_n, 1)
                nms_per_img_ms = nms_batch_ms / max(batch_n, 1)

                for i, det in enumerate(pred):
                    seen += 1
                    p = Path(paths_buf[i])
                    im0 = im0s_buf[i]
                    s = f"{p.name}: "
                    save_path = str(save_dir / p.name)
                    txt_path = str(save_dir / "labels" / p.stem)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    imc = im0.copy() if save_crop else im0
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    if len(det):
                        det[:, :4] = scale_boxes(im_batch.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                            confidence_str = f"{float(conf):.2f}"

                            if save_csv:
                                write_to_csv(p.name, names[c], confidence_str)

                            if save_txt:
                                coords = (
                                    (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    if save_format == 0
                                    else (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                                )
                                line = (cls, *coords, conf) if save_conf else (cls, *coords)
                                with open(f"{txt_path}.txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

                            if save_img or save_crop or view_img:
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(
                                    xyxy,
                                    imc,
                                    file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                                    BGR=True,
                                )

                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == "Linux" and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)
                    if save_img:
                        cv2.imwrite(save_path, im0)

                    # Log por imagem: tempos médios do batch (inferência e NMS) por imagem
                    LOGGER.info(
                        f"{s}{'' if len(det) else '(no detections), '}"
                        f"inf {inf_per_img_ms:.1f}ms/img, nms {nms_per_img_ms:.1f}ms/img"
                    )

                # Resumo do batch: tempos totais e médias por imagem
                LOGGER.info(
                    f"Batch {batch_n} imgs: "
                    f"inference {inf_batch_ms:.1f}ms total ({inf_per_img_ms:.1f}ms/img), "
                    f"NMS {nms_batch_ms:.1f}ms total ({nms_per_img_ms:.1f}ms/img)"
                )

                images_buf.clear()
                im0s_buf.clear()
                paths_buf.clear()
        else:
            # ----------------------------------------------------------
            # 🔁 MODO ORIGINAL (vídeo, webcam, batch=1) - mantive igual
            # ----------------------------------------------------------
            for path, im, im0s, vid_cap, s in dataset:
                # use preprocess_image unificado, mas chamando diretamente (não em executor)
                with dt[0]:
                    im = preprocess_image(im)  # [MOD] unifica pré-process

                with dt[1]:
                    visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize_path)

                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # pós-processamento padrão (inalterado)
                batch_n = len(pred) if isinstance(pred, (list, tuple)) else 1
                inf_batch_ms = dt[1].dt * 1e3
                nms_batch_ms = dt[2].dt * 1e3
                inf_per_img_ms = inf_batch_ms / max(batch_n, 1)
                nms_per_img_ms = nms_batch_ms / max(batch_n, 1)

                for i, det in enumerate(pred):
                    seen += 1
                    if webcam:
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f"{i}: "
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                    p = Path(p)
                    save_path = str(save_dir / p.name)
                    txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
                    s += "{:g}x{:g} ".format(*im.shape[2:])
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    imc = im0.copy() if save_crop else im0
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    if len(det):
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                            confidence_str = f"{float(conf):.2f}"
                            if save_csv:
                                write_to_csv(p.name, names[c], confidence_str)
                            if save_txt:
                                coords = (
                                    (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    if save_format == 0
                                    else (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                                )
                                line = (cls, *coords, conf) if save_conf else (cls, *coords)
                                with open(f"{txt_path}.txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")
                            if save_img or save_crop or view_img:
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == "Linux" and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)

                    if save_img:
                        if dataset.mode == "image":
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path[i] != save_path:
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()
                                if vid_cap:
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix(".mp4"))
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            vid_writer[i].write(im0)

                    # Log por imagem: tempos médios do batch (inferência e NMS) por imagem
                    LOGGER.info(
                        f"{s}{'' if len(det) else '(no detections), '}"
                        f"inf {inf_per_img_ms:.1f}ms/img, nms {nms_per_img_ms:.1f}ms/img"
                    )

                # Resumo do batch: tempos totais e médias por imagem
                LOGGER.info(
                    f"Batch {batch_n} imgs: "
                    f"inference {inf_batch_ms:.1f}ms total ({inf_per_img_ms:.1f}ms/img), "
                    f"NMS {nms_batch_ms:.1f}ms total ({nms_per_img_ms:.1f}ms/img)"
                )

    finally:
        # Garantir limpeza de recursos caso ocorra exceção
        try:
            if producer_thread is not None and producer_thread.is_alive():
                # sinalizar final se necessário (fila será esvaziada)
                pass
        except Exception:
            pass
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass

    # ----------------------------------------------------------
    # 📊 RELATÓRIO FINAL (inalterado)
    # ----------------------------------------------------------
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--workers", type=int, default=4, help="number of dataloader workers")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)